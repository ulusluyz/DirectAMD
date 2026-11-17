#include "directamd/rope.hpp"
#include "polaris_binary.hpp"

#include <numbers>

namespace DirectAMD {

bool rope_fp32(Device& device, uint32_t context, VAManager& va,
               std::span<const float> input, const RopeConfig& config,
               std::span<float> output, std::error_code& ec) {
    if (!RopeKernelParams::valid_shape(input.size(), config.rotary_dim) ||
        !RopeKernelParams::valid_layout(config.layout) || output.size() != input.size() ||
        !std::isfinite(config.frequency_base) || config.frequency_base < 1.0f ||
        !std::ranges::all_of(input, [](float value) { return std::isfinite(value); })) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }
    if (device.get_properties().arch_family != ArchitectureFamily::GCN4_POLARIS) {
        ec = ErrorCode::UNSUPPORTED_GPU;
        return false;
    }

    // Reduce angles on the host before rounding to FP32. The GPU's sine and
    // cosine instructions accept turns, so the table lies in [-0.5, 0.5].
    constexpr long double full_turn = 2.0L * std::numbers::pi_v<long double>;
    const long double log_base = std::log(static_cast<long double>(config.frequency_base));
    std::vector<float> phases(config.rotary_dim / 2);
    for (size_t pair = 0; pair < phases.size(); ++pair) {
        const long double exponent = -static_cast<long double>(pair) / phases.size();
        const long double angle = config.position * std::exp(exponent * log_base);
        phases[pair] = static_cast<float>(std::remainder(angle, full_turn) / full_turn);
    }

    return detail::run_polaris_binary(device, context, va, input, phases, output, 1,
        ISA::PolarisEncoder::emit_rope_kernel(),
        [&](PM4Builder& pm4, uint64_t program, uint64_t x, uint64_t phase, uint64_t y) {
            pm4.add_polaris_rope(program, {
                x, phase, y, static_cast<uint32_t>(input.size()), config.rotary_dim, config.layout
            });
        }, ec, true);
}

} // namespace DirectAMD

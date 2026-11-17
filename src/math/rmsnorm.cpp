#include "directamd/rmsnorm.hpp"
#include "polaris_binary.hpp"

namespace DirectAMD {

bool rmsnorm_fp32(Device& device, uint32_t context, VAManager& va,
                  std::span<const float> input, std::span<const float> weights,
                  float epsilon, std::span<float> output, std::error_code& ec) {
    const auto is_finite = [](float value) { return std::isfinite(value); };
    if (!RmsNormKernelParams::valid_count(input.size()) || weights.size() != input.size() ||
        output.size() != input.size() || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        !std::ranges::all_of(input, is_finite) || !std::ranges::all_of(weights, is_finite)) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    return detail::run_polaris_binary(device, context, va, input, weights, output, 1,
        ISA::PolarisEncoder::emit_rmsnorm_kernel(),
        [&](PM4Builder& pm4, uint64_t program, uint64_t x, uint64_t weight, uint64_t y) {
            pm4.add_polaris_rmsnorm(program, {x, weight, y, static_cast<uint32_t>(input.size()), epsilon});
        }, ec, true);
}

} // namespace DirectAMD

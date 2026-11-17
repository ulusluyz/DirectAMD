#include "directamd/softmax.hpp"
#include "polaris_unary.hpp"

#include <cmath>

namespace DirectAMD {

bool softmax_fp32(Device& device, uint32_t context, VAManager& va,
                  std::span<const float> input, std::span<float> output,
                  std::error_code& ec) {
    if (!SoftmaxKernelParams::valid_count(input.size()) || output.size() != input.size() ||
        !std::ranges::all_of(input, [](float value) { return std::isfinite(value); })) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    return detail::run_polaris_unary(device, context, va, input, output,
        ISA::PolarisEncoder::emit_softmax_kernel(),
        [&](PM4Builder& pm4, uint64_t program, uint64_t x, uint64_t y) {
            pm4.add_polaris_softmax(program, {x, y, static_cast<uint32_t>(input.size())});
        }, [](float p) { return p >= 0.0f && p <= 1.0f; }, ec);
}

} // namespace DirectAMD

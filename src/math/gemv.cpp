#include "directamd/gemv.hpp"
#include "polaris_binary.hpp"

namespace DirectAMD {

bool gemv_fp32(Device& device, uint32_t context, VAManager& va,
               std::span<const float> matrix, std::span<const float> x,
               std::span<float> y, std::error_code& ec) {
    if (!GemvKernelParams::valid_shape(y.size(), x.size()) || matrix.size() != y.size() * x.size()) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    return detail::run_polaris_binary(device, context, va, matrix, x, y, 1,
        ISA::PolarisEncoder::emit_gemv_kernel(),
        [&](PM4Builder& pm4, uint64_t program, uint64_t a, uint64_t b, uint64_t c) {
            pm4.add_polaris_gemv(program, {a, b, c, static_cast<uint32_t>(y.size()),
                                         static_cast<uint32_t>(x.size())});
        }, ec);
}

} // namespace DirectAMD

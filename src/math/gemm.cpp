#include "directamd/gemm.hpp"
#include "polaris_binary.hpp"

namespace DirectAMD {

bool gemm_fp32(Device& device, uint32_t context, VAManager& va,
               uint32_t rows, uint32_t columns, uint32_t inner,
               std::span<const float> a, std::span<const float> b,
               std::span<float> c, std::error_code& ec) {
    if (!GemmKernelParams::valid_shape(rows, columns, inner) ||
        a.size() != static_cast<size_t>(rows) * inner ||
        b.size() != static_cast<size_t>(inner) * columns ||
        c.size() != static_cast<size_t>(rows) * columns) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    return detail::run_polaris_binary(device, context, va, a, b, c, columns,
        ISA::PolarisEncoder::emit_gemm_kernel(),
        [&](PM4Builder& pm4, uint64_t program, uint64_t a_va, uint64_t b_va, uint64_t c_va) {
            pm4.add_polaris_gemm(program, {a_va, b_va, c_va, rows, columns, inner});
        }, ec);
}

} // namespace DirectAMD

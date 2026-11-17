#include "directamd/vector_mul.hpp"
#include "polaris_binary.hpp"

namespace DirectAMD {

bool vector_mul_fp32(Device& device, uint32_t context, VAManager& va,
                     std::span<const float> a, std::span<const float> b,
                     std::span<float> c, std::error_code& ec) {
    const auto finite = [](float value) { return std::isfinite(value); };
    if (!VectorMulKernelParams::valid_count(a.size()) || a.size() != b.size() || a.size() != c.size() ||
        !std::ranges::all_of(a, finite) || !std::ranges::all_of(b, finite)) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    return detail::run_polaris_binary(device, context, va, a, b, c, 1,
        ISA::PolarisEncoder::emit_vector_mul_kernel(),
        [&](PM4Builder& pm4, uint64_t program, uint64_t a_va, uint64_t b_va, uint64_t c_va) {
            pm4.add_polaris_vector_mul(program, {a_va, b_va, c_va, static_cast<uint32_t>(a.size())});
        }, ec, true);
}

} // namespace DirectAMD

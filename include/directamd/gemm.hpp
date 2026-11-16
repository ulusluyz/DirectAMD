#ifndef DIRECTAMD_GEMM_HPP
#define DIRECTAMD_GEMM_HPP

#include <cstdint>
#include <span>
#include <system_error>

namespace DirectAMD {

class Device;
class VAManager;

// Computes C = A * B on Polaris for contiguous row-major FP32 matrices.
// A is rows x inner, B is inner x columns, and C is rows x columns.
// The caller keeps the device, context and VA manager alive.
// Returns after the compute fence and readback. On failure, C is left untouched.
bool gemm_fp32(Device& device, uint32_t context, VAManager& va,
               uint32_t rows, uint32_t columns, uint32_t inner,
               std::span<const float> a, std::span<const float> b,
               std::span<float> c, std::error_code& ec);

} // namespace DirectAMD

#endif // DIRECTAMD_GEMM_HPP

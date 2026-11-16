#ifndef DIRECTAMD_GEMV_HPP
#define DIRECTAMD_GEMV_HPP

#include <cstdint>
#include <span>
#include <system_error>

namespace DirectAMD {

class Device;
class VAManager;

// Computes y = A * x on Polaris. A is contiguous row-major with y.size() rows
// and x.size() columns. The caller keeps the device, context and VA manager alive.
// Returns after the compute fence and readback. On failure, y is left untouched.
bool gemv_fp32(Device& device, uint32_t context, VAManager& va,
               std::span<const float> matrix, std::span<const float> x,
               std::span<float> y, std::error_code& ec);

} // namespace DirectAMD

#endif // DIRECTAMD_GEMV_HPP

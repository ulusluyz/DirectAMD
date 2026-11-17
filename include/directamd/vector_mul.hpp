#ifndef DIRECTAMD_VECTOR_MUL_HPP
#define DIRECTAMD_VECTOR_MUL_HPP

#include <cstdint>
#include <span>
#include <system_error>

namespace DirectAMD {

class Device;
class VAManager;

// Computes c[i] = a[i] * b[i] on Polaris in FP32. The three spans must have the
// same nonzero length and inputs must be finite. FP32 overflow produces signed
// infinity, and subnormals and signed zeros follow FP32 rounding.
// The output may alias either input. The call waits for the GPU fence before
// copying the result, and leaves the output untouched on failure.
bool vector_mul_fp32(Device& device, uint32_t context, VAManager& va,
                     std::span<const float> a, std::span<const float> b,
                     std::span<float> c, std::error_code& ec);

} // namespace DirectAMD

#endif // DIRECTAMD_VECTOR_MUL_HPP

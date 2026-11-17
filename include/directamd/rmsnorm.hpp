#ifndef DIRECTAMD_RMSNORM_HPP
#define DIRECTAMD_RMSNORM_HPP

#include <cstdint>
#include <span>
#include <system_error>

namespace DirectAMD {

class Device;
class VAManager;

// Computes output[i] = input[i] * rsqrt(mean(input^2) + epsilon) * weight[i].
// The three spans must have the same nonzero length. Inputs and weights must
// be finite, and epsilon must be positive and finite.
// The caller owns the device, context and VA manager. The operation waits for
// the compute fence before copying the result. On failure, output is untouched.
bool rmsnorm_fp32(Device& device, uint32_t context, VAManager& va,
                  std::span<const float> input, std::span<const float> weights,
                  float epsilon, std::span<float> output, std::error_code& ec);

} // namespace DirectAMD

#endif // DIRECTAMD_RMSNORM_HPP

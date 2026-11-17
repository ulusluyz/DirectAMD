#ifndef DIRECTAMD_SILU_HPP
#define DIRECTAMD_SILU_HPP

#include <cstdint>
#include <span>
#include <system_error>

namespace DirectAMD {

class Device;
class VAManager;

// Computes output[i] = input[i] / (1 + exp(-input[i])) on Polaris.
// Spans must have the same nonzero length and input values must be finite.
// Input may alias output. The call waits for the GPU fence before copying the
// result, and leaves the output untouched on failure.
bool silu_fp32(Device& device, uint32_t context, VAManager& va,
               std::span<const float> input, std::span<float> output,
               std::error_code& ec);

} // namespace DirectAMD

#endif // DIRECTAMD_SILU_HPP

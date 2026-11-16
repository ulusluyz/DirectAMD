#ifndef DIRECTAMD_SOFTMAX_HPP
#define DIRECTAMD_SOFTMAX_HPP

#include <cstdint>
#include <span>
#include <system_error>

namespace DirectAMD {

class Device;
class VAManager;

// Computes Softmax over one nonempty vector of finite FP32 logits on Polaris.
// Input and output must have the same length. The caller owns the context and VA manager.
// Waits for the GPU fence before copying to output. On failure, output is untouched.
bool softmax_fp32(Device& device, uint32_t context, VAManager& va,
                  std::span<const float> input, std::span<float> output,
                  std::error_code& ec);

} // namespace DirectAMD

#endif // DIRECTAMD_SOFTMAX_HPP

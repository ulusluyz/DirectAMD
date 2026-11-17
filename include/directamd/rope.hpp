#ifndef DIRECTAMD_ROPE_HPP
#define DIRECTAMD_ROPE_HPP

#include "directamd/common.hpp"

#include <span>
#include <system_error>

namespace DirectAMD {

class Device;
class VAManager;

struct RopeConfig {
    uint32_t position{0};
    uint32_t rotary_dim{0};
    float frequency_base{10000.0f};
    RopeLayout layout{RopeLayout::Interleaved};
};

// Rotates the first rotary_dim elements of one FP32 vector on Polaris.
// Pair i rotates by position * frequency_base^(-2*i/rotary_dim) radians.
// Interleaved pairs (2*i, 2*i+1); HalfSplit pairs (i, i+rotary_dim/2).
// Input and output lengths must match. The input must be finite, rotary_dim
// must be positive and even, and frequency_base must be finite and >= 1.
// The kernel copies the suffix unchanged. The call waits for the compute
// fence, and on failure leaves output untouched.
bool rope_fp32(Device& device, uint32_t context, VAManager& va,
               std::span<const float> input, const RopeConfig& config,
               std::span<float> output, std::error_code& ec);

} // namespace DirectAMD

#endif // DIRECTAMD_ROPE_HPP

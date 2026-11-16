#include "directamd/softmax.hpp"

#include "directamd/command.hpp"
#include "directamd/device.hpp"
#include "directamd/isa.hpp"
#include "directamd/memory.hpp"

#include <algorithm>
#include <cmath>
#include <drm/amdgpu_drm.h>
#include <limits>
#include <vector>

namespace DirectAMD {

bool softmax_fp32(Device& device, uint32_t context, VAManager& va,
                  std::span<const float> input, std::span<float> output,
                  std::error_code& ec) {
    if (!SoftmaxKernelParams::valid_count(input.size()) || output.size() != input.size() ||
        !std::ranges::all_of(input, [](float value) { return std::isfinite(value); })) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }
    if (device.get_properties().arch_family != ArchitectureFamily::GCN4_POLARIS) {
        ec = ErrorCode::UNSUPPORTED_GPU;
        return false;
    }

    constexpr size_t guard_count = ISA::PolarisEncoder::wave_size;
    constexpr size_t guard_bytes = guard_count * sizeof(float);
    constexpr float poison = std::numeric_limits<float>::quiet_NaN();
    constexpr float sentinel = -123456.0f;
    const size_t guarded_count = input.size() + 2 * guard_count;
    const size_t guarded_bytes = guarded_count * sizeof(float);
    const auto make_buffer = [&](size_t bytes) {
        return BufferObject::create(device, va, bytes, AMDGPU_GEM_DOMAIN_GTT, ec);
    };

    std::vector<float> staging(guarded_count, poison);
    auto values = std::span(staging).subspan(guard_count, input.size());
    std::ranges::copy(input, values.begin());
    auto bo_input = make_buffer(guarded_bytes);
    if (!bo_input) return false;
    bo_input->copy_to_gpu(staging.data(), guarded_bytes, ec);
    if (ec) return false;

    std::ranges::fill(staging, sentinel);
    std::ranges::fill(values, poison);
    auto bo_output = make_buffer(guarded_bytes);
    if (!bo_output) return false;
    bo_output->copy_to_gpu(staging.data(), guarded_bytes, ec);
    if (ec) return false;

    const auto code = ISA::PolarisEncoder::emit_softmax_kernel();
    const size_t code_bytes = code.size() * sizeof(uint32_t);
    auto bo_code = make_buffer(code_bytes);
    if (!bo_code) return false;
    bo_code->copy_to_gpu(code.data(), code_bytes, ec);
    if (ec) return false;

    PM4Builder pm4;
    pm4.add_polaris_softmax(bo_code->get_gpu_va(), {
        bo_input->get_gpu_va() + guard_bytes, bo_output->get_gpu_va() + guard_bytes,
        static_cast<uint32_t>(input.size())
    });
    auto bo_ib = make_buffer(pm4.get_size_bytes());
    if (!bo_ib) return false;
    bo_ib->copy_to_gpu(pm4.get_packets().data(), pm4.get_size_bytes(), ec);
    if (ec) return false;

    const uint64_t fence = CommandSubmission::submit_compute(
        device, context, *bo_ib, {bo_input.get(), bo_output.get(), bo_code.get()}, ec);
    if (ec) return false;
    if (fence == 0) {
        ec = ErrorCode::COMMAND_SUBMISSION_FAILED;
        return false;
    }

    constexpr uint64_t timeout_ns = 2'000'000'000;
    if (!CommandSubmission::wait_fence(device, context, fence, timeout_ns, ec)) return false;
    bo_output->copy_from_gpu(staging.data(), guarded_bytes, ec);
    if (ec) return false;

    const auto is_sentinel = [](float value) { return value == sentinel; };
    const std::span<const float> result(staging);
    if (!std::ranges::all_of(result.first(guard_count), is_sentinel) ||
        !std::ranges::all_of(result.last(guard_count), is_sentinel) ||
        !std::ranges::all_of(values, [](float p) { return p >= 0.0f && p <= 1.0f; })) {
        ec = ErrorCode::KERNEL_EXECUTION_FAILED;
        return false;
    }

    std::ranges::copy(values, output.begin());
    return true;
}

} // namespace DirectAMD

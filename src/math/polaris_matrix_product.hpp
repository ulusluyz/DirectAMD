#ifndef DIRECTAMD_POLARIS_MATRIX_PRODUCT_HPP
#define DIRECTAMD_POLARIS_MATRIX_PRODUCT_HPP

#include "directamd/command.hpp"
#include "directamd/device.hpp"
#include "directamd/isa.hpp"
#include "directamd/memory.hpp"

#include <algorithm>
#include <drm/amdgpu_drm.h>
#include <limits>
#include <span>
#include <vector>

namespace DirectAMD::detail {

template <typename ConfigureDispatch>
bool run_polaris_matrix_product(Device& device, uint32_t context, VAManager& va,
                                std::span<const float> a, std::span<const float> b,
                                std::span<float> c, size_t b_row_size,
                                const std::vector<uint32_t>& code, ConfigureDispatch configure_dispatch,
                                std::error_code& ec) {
    if (device.get_properties().arch_family != ArchitectureFamily::GCN4_POLARIS) {
        ec = ErrorCode::UNSUPPORTED_GPU;
        return false;
    }

    const auto make_buffer = [&](size_t bytes) {
        return BufferObject::create(device, va, bytes, AMDGPU_GEM_DOMAIN_GTT, ec);
    };
    auto bo_a = make_buffer(a.size_bytes());
    if (!bo_a) return false;
    bo_a->copy_to_gpu(a.data(), a.size_bytes(), ec);
    if (ec) return false;

    constexpr size_t guard_count = ISA::PolarisEncoder::wave_size;
    constexpr float poison = std::numeric_limits<float>::quiet_NaN();
    // A read outside the reduction dimension encounters a full row of NaNs.
    const size_t b_guard_count = std::max(guard_count, b_row_size);
    std::vector<float> guarded_b(b.size() + 2 * b_guard_count, poison);
    std::ranges::copy(b, guarded_b.begin() + b_guard_count);
    const size_t guarded_b_bytes = guarded_b.size() * sizeof(float);
    auto bo_b = make_buffer(guarded_b_bytes);
    if (!bo_b) return false;
    bo_b->copy_to_gpu(guarded_b.data(), guarded_b_bytes, ec);
    if (ec) return false;

    // One wave of sentinels on each side covers any stray lanes in the final workgroup.
    constexpr float sentinel = -123456.0f;
    std::vector<float> readback(c.size() + 2 * guard_count, sentinel);
    auto values = std::span(readback).subspan(guard_count, c.size());
    std::ranges::fill(values, poison);
    const size_t readback_bytes = readback.size() * sizeof(float);
    auto bo_c = make_buffer(readback_bytes);
    if (!bo_c) return false;
    bo_c->copy_to_gpu(readback.data(), readback_bytes, ec);
    if (ec) return false;

    const size_t code_bytes = code.size() * sizeof(uint32_t);
    auto bo_code = make_buffer(code_bytes);
    if (!bo_code) return false;
    bo_code->copy_to_gpu(code.data(), code_bytes, ec);
    if (ec) return false;

    PM4Builder pm4;
    configure_dispatch(pm4, bo_code->get_gpu_va(), bo_a->get_gpu_va(),
                       bo_b->get_gpu_va() + b_guard_count * sizeof(float),
                       bo_c->get_gpu_va() + guard_count * sizeof(float));
    auto bo_ib = make_buffer(pm4.get_size_bytes());
    if (!bo_ib) return false;
    bo_ib->copy_to_gpu(pm4.get_packets().data(), pm4.get_size_bytes(), ec);
    if (ec) return false;

    const uint64_t fence = CommandSubmission::submit_compute(
        device, context, *bo_ib, {bo_a.get(), bo_b.get(), bo_c.get(), bo_code.get()}, ec);
    if (ec) return false;
    if (fence == 0) {
        ec = ErrorCode::COMMAND_SUBMISSION_FAILED;
        return false;
    }

    constexpr uint64_t timeout_ns = 2'000'000'000;
    if (!CommandSubmission::wait_fence(device, context, fence, timeout_ns, ec)) return false;
    bo_c->copy_from_gpu(readback.data(), readback_bytes, ec);
    if (ec) return false;

    const auto is_sentinel = [=](float value) { return value == sentinel; };
    const std::span<const float> result(readback);
    if (!std::ranges::all_of(result.first(guard_count), is_sentinel) ||
        !std::ranges::all_of(result.last(guard_count), is_sentinel)) {
        ec = ErrorCode::KERNEL_EXECUTION_FAILED;
        return false;
    }

    std::ranges::copy(values, c.begin());
    return true;
}

} // namespace DirectAMD::detail

#endif // DIRECTAMD_POLARIS_MATRIX_PRODUCT_HPP

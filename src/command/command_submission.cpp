#include "directamd/command.hpp"
#include "directamd/device.hpp"
#include "directamd/memory.hpp"

#include <cerrno>
#include <limits>
#include <time.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <drm/drm.h>
#include <drm/amdgpu_drm.h>

namespace DirectAMD {

uint64_t CommandSubmission::submit_compute(Device& device, uint32_t ctx_id,
                                           BufferObject& ib_bo,
                                           const std::vector<BufferObject*>& bo_handles,
                                           std::error_code& ec) {
    int fd = device.get_fd();
    if (fd < 0) {
        ec = make_error_code(ErrorCode::DEVICE_NOT_FOUND);
        return 0;
    }

    const size_t ib_bytes = ib_bo.get_requested_size();
    if (ib_bytes == 0 || ib_bytes > std::numeric_limits<uint32_t>::max() ||
        ib_bytes % sizeof(uint32_t) != 0) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return 0;
    }

    // 1. Create BO list via DRM_IOCTL_AMDGPU_BO_LIST
    std::vector<struct drm_amdgpu_bo_list_entry> bo_entries;
    bo_entries.reserve(bo_handles.size() + 1);

    struct drm_amdgpu_bo_list_entry ib_entry{};
    ib_entry.bo_handle = ib_bo.get_handle();
    ib_entry.bo_priority = 0;
    bo_entries.push_back(ib_entry);

    for (auto* bo : bo_handles) {
        if (bo) {
            struct drm_amdgpu_bo_list_entry entry{};
            entry.bo_handle = bo->get_handle();
            entry.bo_priority = 0;
            bo_entries.push_back(entry);
        }
    }

    union drm_amdgpu_bo_list bo_list_args{};
    bo_list_args.in.operation = AMDGPU_BO_LIST_OP_CREATE;
    bo_list_args.in.list_handle = 0;
    bo_list_args.in.bo_number = static_cast<uint32_t>(bo_entries.size());
    bo_list_args.in.bo_info_size = sizeof(struct drm_amdgpu_bo_list_entry);
    bo_list_args.in.bo_info_ptr = reinterpret_cast<uint64_t>(bo_entries.data());

    if (ioctl(fd, DRM_IOCTL_AMDGPU_BO_LIST, &bo_list_args) < 0) {
        ec = std::error_code(errno, std::generic_category());
        return 0;
    }

    uint32_t bo_list_handle = bo_list_args.out.list_handle;

    // 2. Prepare Indirect Buffer chunk
    struct drm_amdgpu_cs_chunk_ib ib_chunk_data{};
    ib_chunk_data.va_start = ib_bo.get_gpu_va();
    ib_chunk_data.ib_bytes = static_cast<uint32_t>(ib_bytes);
    ib_chunk_data.ip_type = AMDGPU_HW_IP_COMPUTE;
    ib_chunk_data.ip_instance = 0;
    ib_chunk_data.ring = 0;
    ib_chunk_data.flags = AMDGPU_IB_FLAG_EMIT_MEM_SYNC;

    struct drm_amdgpu_cs_chunk chunk{};
    chunk.chunk_id = AMDGPU_CHUNK_ID_IB;
    chunk.length_dw = sizeof(struct drm_amdgpu_cs_chunk_ib) / 4;
    chunk.chunk_data = reinterpret_cast<uint64_t>(&ib_chunk_data);

    union drm_amdgpu_cs cs_args{};
    cs_args.in.ctx_id = ctx_id;
    cs_args.in.bo_list_handle = bo_list_handle;
    cs_args.in.num_chunks = 1;
    // DRM expects an array of userspace addresses, one per chunk.
    uint64_t chunk_addresses[] = {reinterpret_cast<uint64_t>(&chunk)};
    cs_args.in.chunks = reinterpret_cast<uint64_t>(chunk_addresses);

    const int result = ioctl(fd, DRM_IOCTL_AMDGPU_CS, &cs_args);
    const int submission_errno = result < 0 ? errno : 0;

    union drm_amdgpu_bo_list destroy_list{};
    destroy_list.in.operation = AMDGPU_BO_LIST_OP_DESTROY;
    destroy_list.in.list_handle = bo_list_handle;
    ioctl(fd, DRM_IOCTL_AMDGPU_BO_LIST, &destroy_list);

    if (result < 0) {
        ec = std::error_code(submission_errno, std::generic_category());
        return 0;
    }

    ec.clear();
    return cs_args.out.handle; // Returned sequence fence number
}

bool CommandSubmission::wait_fence(Device& device, uint32_t ctx_id, uint64_t fence_handle,
                                  uint64_t timeout_ns, std::error_code& ec) {
    int fd = device.get_fd();
    if (fd < 0) {
        ec = make_error_code(ErrorCode::DEVICE_NOT_FOUND);
        return false;
    }

    // WAIT_CS takes an absolute CLOCK_MONOTONIC deadline in nanoseconds.
    struct timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) < 0) {
        ec = std::error_code(errno, std::generic_category());
        return false;
    }
    const uint64_t now_ns = static_cast<uint64_t>(now.tv_sec) * 1000000000ULL + now.tv_nsec;
    const uint64_t max_timeout = std::numeric_limits<int64_t>::max();

    union drm_amdgpu_wait_cs wait_args{};
    wait_args.in.handle = fence_handle;
    wait_args.in.timeout = timeout_ns > max_timeout - now_ns ? max_timeout : now_ns + timeout_ns;
    wait_args.in.ip_type = AMDGPU_HW_IP_COMPUTE;
    wait_args.in.ip_instance = 0;
    wait_args.in.ring = 0;
    wait_args.in.ctx_id = ctx_id;

    if (ioctl(fd, DRM_IOCTL_AMDGPU_WAIT_CS, &wait_args) < 0) {
        ec = std::error_code(errno, std::generic_category());
        return false;
    }
    if (wait_args.out.status != 0) {
        ec = make_error_code(ErrorCode::GPU_TIMEOUT);
        return false;
    }

    ec.clear();
    return true;
}

} // namespace DirectAMD

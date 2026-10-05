#include "directamd/memory.hpp"
#include "directamd/device.hpp"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <drm/drm.h>
#include <drm/amdgpu_drm.h>

namespace DirectAMD {

std::unique_ptr<BufferObject> BufferObject::create(Device& device, VAManager& va_mgr,
                                                   size_t size_bytes, uint32_t domain_flags,
                                                   std::error_code& ec) {
    int fd = device.get_fd();
    if (fd < 0) {
        ec = make_error_code(ErrorCode::DEVICE_NOT_FOUND);
        return nullptr;
    }

    // Ensure 4KB alignment
    size_t page_aligned_size = (size_bytes + 4095) & ~4095;

    // 1. GEM BO Allocation
    union drm_amdgpu_gem_create alloc_args{};
    alloc_args.in.bo_size = page_aligned_size;
    alloc_args.in.alignment = 4096;
    alloc_args.in.domains = domain_flags;
    alloc_args.in.domain_flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED;

    if (ioctl(fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &alloc_args) < 0) {
        ec = make_error_code(ErrorCode::MEMORY_ALLOCATION_FAILED);
        return nullptr;
    }

    uint32_t handle = alloc_args.out.handle;

    // 2. Map CPU Host virtual address
    union drm_amdgpu_gem_mmap mmap_args{};
    mmap_args.in.handle = handle;
    void* cpu_ptr = nullptr;

    if (ioctl(fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &mmap_args) == 0) {
        cpu_ptr = mmap(NULL, page_aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                       fd, mmap_args.out.addr_ptr);
        if (cpu_ptr == MAP_FAILED) {
            cpu_ptr = nullptr;
        }
    }

    // 3. Allocate and map GPU Virtual Address (48-bit)
    uint64_t gpu_va = va_mgr.allocate_range(page_aligned_size, 4096);

    struct drm_amdgpu_gem_va va_args{};
    va_args.handle = handle;
    va_args.operation = AMDGPU_VA_OP_MAP;
    va_args.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE;
    va_args.va_address = gpu_va;
    va_args.offset_in_bo = 0;
    va_args.map_size = page_aligned_size;

    if (ioctl(fd, DRM_IOCTL_AMDGPU_GEM_VA, &va_args) < 0) {
        if (cpu_ptr) {
            munmap(cpu_ptr, page_aligned_size);
        }
        struct drm_gem_close close_args{handle, 0};
        ioctl(fd, DRM_IOCTL_GEM_CLOSE, &close_args);
        ec = make_error_code(ErrorCode::GPU_MAPPING_FAILED);
        return nullptr;
    }

    ec.clear();
    return std::unique_ptr<BufferObject>(
        new BufferObject(device, va_mgr, handle, page_aligned_size, gpu_va, cpu_ptr));
}

BufferObject::BufferObject(Device& device, VAManager& va_mgr, uint32_t handle,
                           size_t size_bytes, uint64_t gpu_va, void* cpu_ptr)
    : device_(device), va_mgr_(va_mgr), gem_handle_(handle), size_bytes_(size_bytes),
      gpu_va_(gpu_va), cpu_ptr_(cpu_ptr) {}

BufferObject::~BufferObject() {
    int fd = device_.get_fd();
    if (fd >= 0) {
        // Unmap GPU Virtual Address
        struct drm_amdgpu_gem_va va_args{};
        va_args.handle = gem_handle_;
        va_args.operation = AMDGPU_VA_OP_UNMAP;
        va_args.flags = 0;
        va_args.va_address = gpu_va_;
        va_args.offset_in_bo = 0;
        va_args.map_size = size_bytes_;
        ioctl(fd, DRM_IOCTL_AMDGPU_GEM_VA, &va_args);

        // Unmap CPU Address
        if (cpu_ptr_) {
            munmap(cpu_ptr_, size_bytes_);
            cpu_ptr_ = nullptr;
        }

        // Close GEM Handle
        struct drm_gem_close close_args{gem_handle_, 0};
        ioctl(fd, DRM_IOCTL_GEM_CLOSE, &close_args);
    }
    va_mgr_.free_range(gpu_va_, size_bytes_);
}

void BufferObject::copy_to_gpu(const void* src, size_t count_bytes, std::error_code& ec) {
    if (count_bytes > size_bytes_) {
        ec = make_error_code(ErrorCode::OUT_OF_MEMORY);
        return;
    }
    if (!cpu_ptr_) {
        ec = make_error_code(ErrorCode::GPU_MAPPING_FAILED);
        return;
    }
    std::memcpy(cpu_ptr_, src, count_bytes);
    ec.clear();
}

void BufferObject::copy_from_gpu(void* dst, size_t count_bytes, std::error_code& ec) {
    if (count_bytes > size_bytes_) {
        ec = make_error_code(ErrorCode::OUT_OF_MEMORY);
        return;
    }
    if (!cpu_ptr_) {
        ec = make_error_code(ErrorCode::GPU_MAPPING_FAILED);
        return;
    }
    std::memcpy(dst, cpu_ptr_, count_bytes);
    ec.clear();
}

} // namespace DirectAMD

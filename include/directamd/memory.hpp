#ifndef DIRECTAMD_MEMORY_HPP
#define DIRECTAMD_MEMORY_HPP

#include "directamd/common.hpp"
#include "directamd/error.hpp"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <atomic>

namespace DirectAMD {

class Device;

class VAManager {
public:
    explicit VAManager(uint64_t start_address = 0x100000000ULL); // Start above 4GB
    ~VAManager() = default;

    uint64_t allocate_range(size_t bytes, size_t alignment = 4096);
    void free_range(uint64_t gpu_va, size_t bytes);

private:
    std::atomic<uint64_t> current_va_;
};

class BufferObject {
public:
    static std::unique_ptr<BufferObject> create(Device& device, VAManager& va_mgr,
                                                size_t size_bytes, uint32_t domain_flags,
                                                std::error_code& ec);

    ~BufferObject();

    BufferObject(const BufferObject&) = delete;
    BufferObject& operator=(const BufferObject&) = delete;

    uint32_t get_handle() const noexcept { return gem_handle_; }
    size_t get_size() const noexcept { return size_bytes_; }
    uint64_t get_gpu_va() const noexcept { return gpu_va_; }
    void* get_cpu_ptr() const noexcept { return cpu_ptr_; }

    void copy_to_gpu(const void* src, size_t count_bytes, std::error_code& ec);
    void copy_from_gpu(void* dst, size_t count_bytes, std::error_code& ec);

private:
    BufferObject(Device& device, VAManager& va_mgr, uint32_t handle, size_t size_bytes,
                 uint64_t gpu_va, void* cpu_ptr);

    Device& device_;
    VAManager& va_mgr_;
    uint32_t gem_handle_{0};
    size_t size_bytes_{0};
    uint64_t gpu_va_{0};
    void* cpu_ptr_{nullptr};
};

} // namespace DirectAMD

#endif // DIRECTAMD_MEMORY_HPP

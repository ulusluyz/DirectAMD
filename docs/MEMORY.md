# DirectAMD Memory Management Specification

## Overview
Low-level memory management on AMD GPUs requires managing physical allocation domains, 48-bit Virtual Address (VA) spaces, and CPU/GPU memory coherent mappings through the Linux DRM AMDGPU kernel UAPI.

---

## 1. Memory Domains

AMD GPU memory buffers are allocated across three distinct domains:

1. **VRAM (`AMDGPU_GEM_DOMAIN_VRAM`):** High-bandwidth local GDDR6/HBM memory located directly on the GPU card. Optimal for weight storage and active intermediate tensor buffers.
2. **GTT (`AMDGPU_GEM_DOMAIN_GTT`):** Graphics Translation Table memory—system RAM made accessible to the GPU PCIe DMA engine. Used for host transfer staging and mapped CPU-visible buffers.
3. **CPU (`AMDGPU_GEM_DOMAIN_CPU`):** Pure host memory not visible to GPU engines.

---

## 2. Allocation & Mapping Workflow

```
1. Allocation
   ioctl(fd, DRM_IOCTL_AMDGPU_GEM_CREATE)
   -> Returns GEM BO Handle

2. Host CPU Mapping (if required)
   ioctl(fd, DRM_IOCTL_AMDGPU_GEM_MMAP)
   mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset)
   -> Returns CPU Host Virtual Address Pointer

3. GPU Virtual Address Allocation & Binding
   allocate_gpu_va_range(size, alignment=4096)
   ioctl(fd, DRM_IOCTL_AMDGPU_GEM_VA, AMDGPU_VA_OP_MAP)
   -> Binds GEM Handle to 48-bit GPU Virtual Address
```

---

## 3. Buffer Alignment & Page Size Constraints

* **Page Size Alignment:** All GPU memory allocations and VA mappings must be aligned to 4096 bytes (4 KB).
* **Pointer Alignment:** 64-bit pointers in AMD ISA instruction loads (`GLOBAL_LOAD_DWORD`) require 32-bit alignment; 128-bit vector loads require 128-bit alignment.
* **Kernel Binary Alignment:** GPU code objects must be aligned to 256 bytes for `COMPUTE_PGM_LO/HI`.

---

## 4. RAII Memory Management Architecture

DirectAMD encapsulates buffer lifetimes in `DirectAMD::Memory::BufferObject`:

```cpp
class BufferObject {
public:
    BufferObject(int fd, size_t size, uint32_t domain, uint64_t gpu_va);
    ~BufferObject(); // Automatically unmaps VA and releases GEM handle

    void* get_cpu_ptr() const;
    uint64_t get_gpu_va() const;
    size_t get_size() const;

    void copy_to_gpu(const void* host_src, size_t bytes);
    void copy_from_gpu(void* host_dst, size_t bytes);
};
```
This guarantees deterministic cleanup and zero memory leaks.

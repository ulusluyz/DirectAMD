# AMDGPU DRM Kernel Driver Research

## Overview
This document details the low-level Linux Kernel AMDGPU DRM UAPI structures, ioctl numbers, memory domain flags, and command submission semantics used by DirectAMD.

---

## 1. Device Discovery & Information Query (`DRM_IOCTL_AMDGPU_INFO`)

The primary mechanism to query GPU attributes from userspace is `DRM_IOCTL_AMDGPU_INFO`.

```c
struct drm_amdgpu_info {
    __u32 return_pointer; // Address of output buffer
    __u32 return_size;    // Size of output buffer
    __u32 query;          // AMDGPU_INFO_* tag
    __u32 flags;
    // ... union query data ...
};
```

### Key Query Tags
* `AMDGPU_INFO_DEV_INFO` (`0x16`): Returns `struct drm_amdgpu_info_device` containing PCI device ID, chip revision, ASIC family (e.g. `FAMILY_NV`), compute unit counts, maximum engine clocks, and VRAM/GTT sizes.
* `AMDGPU_INFO_VRAM_GTT` (`0x08`): Returns VRAM and GTT total/usable sizes and un-allocatable memory holes.
* `AMDGPU_INFO_MEMORY` (`0x19`): Detailed memory budget and heap usage info.
* `AMDGPU_INFO_HW_IP_INFO` (`0x02`): Information about specific hardware rings (`AMDGPU_HW_IP_COMPUTE`, `AMDGPU_HW_IP_GFX`, `AMDGPU_HW_IP_DMA`).

---

## 2. Context Creation (`DRM_IOCTL_AMDGPU_CTX`)

Before submitting command buffers, a GPU context must be created:

```c
union drm_amdgpu_ctx {
    struct drm_amdgpu_ctx_in {
        __u32 op;         // AMDGPU_CTX_OP_ALLOC_CTX / AMDGPU_CTX_OP_FREE_CTX
        __u32 flags;
        __u32 ctx_id;     // Returned context ID
        __u32 priority;   // AMDGPU_CTX_PRIORITY_NORMAL
    } in;
    struct drm_amdgpu_ctx_out {
        __u32 alloc_ctx_id;
    } out;
};
```

An allocated `ctx_id` represents a virtualized hardware state on the GPU Command Processor (CP).

---

## 3. Buffer Object Management & Virtual Address Space

### Buffer Allocation (`DRM_IOCTL_AMDGPU_GEM_CREATE`)
Memory on AMD GPUs is allocated via GEM Buffer Objects (BOs):

```c
union drm_amdgpu_gem_create {
    struct {
        __u64 alloc_size;     // Size in bytes (page aligned)
        __u64 alignment;      // Alignment in bytes
        __u64 domains;        // AMDGPU_GEM_DOMAIN_VRAM or AMDGPU_GEM_DOMAIN_GTT
        __u64 flags;          // AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED, AMDGPU_GEM_CREATE_NO_CPU_ACCESS
    } in;
    struct {
        __u32 handle;         // GEM handle
        __u32 _pad;
    } out;
};
```

### CPU Memory Mapping (`DRM_IOCTL_AMDGPU_GEM_MMAP`)
To allow host CPU reads/writes on a GEM handle, an mmap offset is requested:

```c
union drm_amdgpu_gem_mmap {
    struct {
        __u32 handle;
        __u32 _pad;
    } in;
    struct {
        __u64 addr_ptr; // Offset to pass to mmap(2) on the render node fd
    } out;
};
```

### GPU Virtual Address Binding (`DRM_IOCTL_AMDGPU_GEM_VA`)
GPU engines execute using 48-bit GPU Virtual Addresses. A GEM BO handle must be mapped into GPU virtual space:

```c
struct drm_amdgpu_gem_va {
    __u32 handle;
    __u32 _pad;
    __u32 operation;   // AMDGPU_VA_OP_MAP / AMDGPU_VA_OP_UNMAP
    __u32 flags;       // AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE
    __u64 va_address;  // Target 64-bit GPU virtual address
    __u64 offset_in_bo;// Offset inside BO
    __u64 map_size;    // Size in bytes
};
```

---

## 4. Command Submission (`DRM_IOCTL_AMDGPU_CS`)

Command submission transfers indirect command buffers containing PM4 packets to the hardware rings.

```c
union drm_amdgpu_cs {
    struct {
        __u32 ctx_id;         // Context ID from DRM_IOCTL_AMDGPU_CTX
        __u32 bo_list_handle; // BO handle list for validation
        __u32 num_chunks;     // Number of submission chunks
        __u32 _pad;
        __u64 chunks;         // Pointer to array of struct drm_amdgpu_cs_chunk
    } in;
    struct {
        __u64 handle;         // Sequence fence number returned by driver
    } out;
};
```

Each chunk specifies either command buffer descriptors (`AMDGPU_CHUNK_ID_IB`), fence dependencies (`AMDGPU_CHUNK_ID_DEPENDENCIES`), or BO lists.

---

## 5. Synchronization (`DRM_IOCTL_AMDGPU_WAIT_CS`)

To check if a submitted command buffer has finished execution:

```c
union drm_amdgpu_wait_cs {
    struct {
        __u64 handle;     // Fence handle returned by CS ioctl
        __u64 timeout_ns; // Timeout in nanoseconds
        __u32 ip_type;    // AMDGPU_HW_IP_COMPUTE
        __u32 ip_instance;
        __u32 ring;
        __u32 ctx_id;
    } in;
    struct {
        __u64 status;     // 0 = complete, non-zero = busy/error
    } out;
};
```

# DirectAMD Architecture Specification

## Overview
DirectAMD is an experimental, standalone C++20 low-level runtime designed for direct execution of compute tasks on AMD GPUs without CUDA, ROCm, HIP, Vulkan, or OpenCL.

```
                    ┌─────────────────────────┐
                    │    Application / C++    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    DirectAMD Runtime    │
                    │  (src/device, memory,   │
                    │   command, isa, etc.)   │
                    └────────────┬────────────┘
                                 │ ioctl()
                    ┌────────────▼────────────┐
                    │ AMDGPU DRM UAPI Kernel  │
                    │   (/dev/dri/renderD*)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Linux amdgpu.ko Driver  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   AMD GPU Hardware      │
                    │  (Compute Ring / CUs)   │
                    └─────────────────────────┘
```

---

## Subsystem Architecture

### 1. Device Subsystem (`src/device/`)
* Discovers DRM render nodes in `/dev/dri/renderD128..191`.
* Opens device node handles (`open(2)`).
* Queries GPU capabilities via `DRM_IOCTL_AMDGPU_INFO`:
  * Chip Family (e.g. `FAMILY_NV`), Device ID, Revision.
  * VRAM size, GTT size, Memory budget.
  * Compute unit count, max wave limits, hardware ring counts.

### 2. Memory Subsystem (`src/memory/`)
* Manages GEM Buffer Objects (BOs) in VRAM (`AMDGPU_GEM_DOMAIN_VRAM`) or GTT (`AMDGPU_GEM_DOMAIN_GTT`).
* Allocates host-accessible Mapped Memory (`DRM_IOCTL_AMDGPU_GEM_MMAP`).
* Manages 48-bit GPU Virtual Address (VA) space via `DRM_IOCTL_AMDGPU_GEM_VA`.
* Tracks allocation handles, host virtual addresses, and GPU virtual addresses for deterministic cleanup using RAII.

### 3. Command Subsystem (`src/command/`)
* Allocates GPU contexts (`DRM_IOCTL_AMDGPU_CTX`).
* Builds PM4 packet buffers targeting the Compute Ring (`AMDGPU_HW_IP_COMPUTE`).
* Serializes dispatch configurations:
  * Program base addresses (`COMPUTE_PGM_LO` / `COMPUTE_PGM_HI`).
  * User SGPR argument registers.
  * Local/Global workgroup dimensions.
  * PM4 `PACKET3_DISPATCH_DIRECT`.
* Submits command buffers to kernel via `DRM_IOCTL_AMDGPU_CS` with associated BO handles for buffer validation.
* Synchronizes completion using fence sequence handles (`DRM_IOCTL_AMDGPU_WAIT_CS`).

### 4. ISA Subsystem (`src/isa/`)
* Generates raw GPU machine instructions directly in C++.
* Modular architecture:
  * `src/isa/common/`: Common instruction traits and register definitions.
  * `src/isa/rdna/`: RDNA2 / RDNA3 binary instruction synthesizer.
  * `src/isa/gcn/`: GCN/Vega instruction synthesizer interface.

---

## Safety & Error Handling Principles
* **No Hidden Fallbacks:** If a GPU call or device node is missing, the system strictly returns explicit device errors (`DEVICE_NOT_FOUND`, `HARDWARE_UNAVAILABLE`). CPU fallback is never passed off as GPU execution.
* **Warning-Clean Build:** Compiled with `-Wall -Wextra -Wpedantic -Werror`.
* **Resource Safety:** Deterministic cleanup via RAII wrappers (`DeviceFd`, `ContextHandle`, `BufferObject`).

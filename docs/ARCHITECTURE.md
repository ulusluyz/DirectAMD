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
* `src/isa/polaris_encoder.cpp` provides the validated wave64 vector-add,
  GEMV and GEMM kernels for `gfx803`.
* `src/isa/rdna_encoder.cpp` contains the RDNA instruction research code.

### 5. FP32 Matrix Operations (`src/math/`)
* `gemv_fp32` and `gemm_fp32` accept contiguous row-major inputs. They check
  positive dimensions, span lengths and the kernel's 32-bit byte-offset
  limits before allocating or submitting buffers.
* The shared Polaris path stages inputs and output in host-visible GTT
  BOs. It submits the code and indirect buffer to the selected device,
  waits for the compute fence and checks output sentinels before copying
  the result to the caller.
* GEMV assigns one row per lane. GEMM uses a two-dimensional grid in
  which workgroup Y selects the row and the wave's X lanes select output
  columns. The GEMM kernel masks partial waves. Both kernels iterate the
  reduction dimension with separate FP32 multiply and add instructions.

---

## Safety & Error Handling Principles
* **No Hidden Fallbacks:** If a GPU call or device node is missing, the system strictly returns explicit device errors (`DEVICE_NOT_FOUND`, `HARDWARE_UNAVAILABLE`). CPU fallback is never passed off as GPU execution.
* **Warning-Clean Build:** Compiled with `-Wall -Wextra -Wpedantic -Werror`.
* **Resource Safety:** Deterministic cleanup via RAII wrappers (`DeviceFd`, `ContextHandle`, `BufferObject`).

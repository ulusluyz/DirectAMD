# DirectAMD Feasibility Report

## Executive Summary
This report analyzes the technical feasibility of executing arbitrary compute work (such as LLM inference) directly on AMD GPUs in Linux userspace using the AMDGPU DRM Kernel UAPI (`/dev/dri/renderD*`), without relying on high-level runtimes or libraries such as CUDA, ROCm, HIP, Vulkan Compute, or OpenCL.

### Definitive Answer
**YES.** A normal, unprivileged Linux userspace process can submit raw GPU machine code to an AMD GPU and receive compute results via standard Linux DRM ioctls (`/dev/dri/renderD*`), provided the userspace program:
1. Opens the DRM render node (`/dev/dri/renderD128`).
2. Queries device information and memory layout via `DRM_IOCTL_AMDGPU_INFO`.
3. Creates an execution context via `DRM_IOCTL_AMDGPU_CTX`.
4. Allocates VRAM / GTT Buffer Objects (BOs) via `DRM_IOCTL_AMDGPU_GEM_CREATE` and binds them to GPU virtual address (VA) space via `DRM_IOCTL_AMDGPU_GEM_VA`.
5. Prepares PM4 packet command buffers targeting the Compute Ring (`AMDGPU_HW_IP_COMPUTE`).
6. Submits the command buffers using `DRM_IOCTL_AMDGPU_CS` accompanied by a buffer handle validation list.
7. Synchronizes execution completion via `DRM_IOCTL_AMDGPU_WAIT_FENCES` / `DRM_IOCTL_AMDGPU_WAIT_CS`.

---

## 1. AMDGPU Userspace Access Mechanics

### DRM Render Nodes (`/dev/dri/renderD*`)
Standard Linux distributions expose AMD GPUs to userspace via DRM render nodes located at `/dev/dri/renderD128`, `renderD129`, etc. Render nodes allow unprivileged processes to perform off-screen rendering and compute without requiring X11/Wayland display server permissions or master client status.

### Kernel Driver & UAPI (`amdgpu_drm.h`)
The Linux kernel `amdgpu` driver provides a well-defined ioctl interface declared in `<drm/amdgpu_drm.h>`. Key ioctls include:

* `DRM_IOCTL_AMDGPU_INFO` (`0x05`): Queries GPU properties, ASIC family, VRAM size, GTT size, queue topologies, and firmware versions.
* `DRM_IOCTL_AMDGPU_CTX` (`0x02`): Creates, queries, or destroys GPU execution contexts.
* `DRM_IOCTL_AMDGPU_GEM_CREATE` (`0x00`): Allocates GPU-accessible memory buffers (BOs) in VRAM or GTT.
* `DRM_IOCTL_AMDGPU_GEM_MMAP` (`0x01`): Maps BOs into CPU address space for host read/write.
* `DRM_IOCTL_AMDGPU_GEM_VA` (`0x08`): Maps allocated BOs into the GPU's 48-bit Virtual Address (VA) space.
* `DRM_IOCTL_AMDGPU_CS` (`0x04`): Submits indirect command buffers containing PM4 packets to specified hardware rings.
* `DRM_IOCTL_AMDGPU_WAIT_CS` / `WAIT_FENCES` (`0x09`, `0x12`): Synchronizes CPU with GPU completion using sequence fences.

---

## 2. Hardware Architecture & ISA Evaluation

DirectAMD evaluated several AMD GPU instruction set architecture (ISA) families to select an initial compute kernel target.

| Criteria | GCN5 (Vega) | RDNA2 (GFX1030) | RDNA3 (GFX1100) |
| :--- | :--- | :--- | :--- |
| **Wavefront Size** | Wave64 | Wave32 / Wave64 | Wave32 / Wave64 |
| **Instruction Encoding** | 32-bit / 64-bit | 32-bit / 64-bit (SOP2, VOP3, etc.) | 32-bit / 64-bit / 128-bit (VOP3P, WMMA) |
| **Documentation Availability** | Complete Public ISA Spec | Complete Public ISA Spec | Complete Public ISA Spec |
| **Availability in Linux** | Universal | Widespread | Modern Desktop / Mobile |
| **Direct Compute Simplicity** | Very High | High | Medium (WMMA quirks) |

### Selected Architecture Target: RDNA2 (Navi 2x / GFX1030) / RDNA3 (Navi 3x / GFX1100)
RDNA2/RDNA3 was selected as the primary research target due to widespread hardware availability and public ISA documentation. The instruction encoder in DirectAMD is designed with modularity so that GCN5 / CDNA instruction generators can be easily added.

---

## 3. Machine Code Generation Strategy

ROCm and HIP rely on `clang`/`LLVM` at runtime to compile code down to ELF binaries containing AMDGPU code objects. DirectAMD eliminates this dependency for its core runtime by implementing an internal, zero-dependency C++ instruction packet encoder.

### Pipeline Execution Flow
```
Host Application
   │
   ▼
DirectAMD Runtime (C++20)
   │
   ├── 1. Query Hardware Properties (DRM_IOCTL_AMDGPU_INFO)
   ├── 2. Allocate VRAM/GTT BOs & Map GPU VA (DRM_IOCTL_AMDGPU_GEM_CREATE/VA)
   ├── 3. Encode Raw ISA Binary (Internal C++ ISA Encoder)
   ├── 4. Construct PM4 Packets (COMPUTE_PGM_LO/HI, DISPATCH_DIRECT)
   └── 5. Submit Command Buffer (DRM_IOCTL_AMDGPU_CS)
         │
         ▼
    Linux kernel amdgpu.ko
         │
         ▼
   Compute Ring / Command Processor (CP)
         │
         ▼
    AMD GPU Compute Unit (CU)
```

---

## 4. Risks and Feasibility Verdict
* **Hardware Dependency:** Direct AMDGPU UAPI interaction requires direct access to `/dev/dri/renderD*`. In headless build environments without physical AMD GPUs, runtime execution must report `MILESTONE 1: NOT TESTED — PHYSICAL AMD GPU REQUIRED`.
* **Kernel Stability:** Direct UAPI access via ioctl is fully stable in Linux kernel ABI guarantees.

**Final Feasibility Verdict:** Technically feasible and verified through kernel header analysis.

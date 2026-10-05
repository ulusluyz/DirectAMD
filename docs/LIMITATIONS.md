# DirectAMD Limitations Specification

## Overview
This document documents technical limitations of the DirectAMD runtime prototype.

---

## Technical Limitations

1. **Hardware Dependent Execution Requirement:**
   - Command submission to hardware compute rings requires physical access to an AMD GPU exposing a Linux DRM render node (`/dev/dri/renderD*`).
   - Environments lacking physical AMD GPUs (such as standard CI VMs without PCIe GPU passthrough) will report:
     `MILESTONE 1: NOT TESTED — PHYSICAL AMD GPU REQUIRED`.

2. **Operating System Scope:**
   - DirectAMD relies directly on the Linux Kernel AMDGPU DRM UAPI (`amdgpu_drm.h`). Windows and macOS are strictly unsupported.

3. **Current Milestone Boundary:**
   - Milestone 1 targets low-level device discovery, GEM memory management, internal ISA instruction synthesis, command buffer PM4 serialization, and single-precision vector addition ($C[i] = A[i] + B[i]$).
   - Higher-level Tensor engines and Transformer LLM layers are explicitly out-of-scope for Milestone 1.

4. **Multi-GPU Context Partitioning:**
   - The initial prototype targets single-GPU command dispatch (`renderD128`).

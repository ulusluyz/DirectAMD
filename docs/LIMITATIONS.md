# DirectAMD Limitations Specification

## Overview
This document documents technical limitations of the DirectAMD runtime prototype.

---

## Technical Limitations

1. **Hardware Dependent Execution Requirement:**
   - Command submission to hardware compute rings requires physical access to an AMD GPU exposing a Linux DRM render node (`/dev/dri/renderD*`).
   - If a compatible render node is unavailable, the GPU example exits nonzero and GPU execution remains untested.

2. **Operating System Scope:**
   - DirectAMD relies directly on the Linux Kernel AMDGPU DRM UAPI (`amdgpu_drm.h`). Windows and macOS are strictly unsupported.

3. **Current Milestone Boundary:**
   - Milestone 1 targets low-level device discovery, GEM memory management, internal ISA instruction synthesis, command buffer PM4 serialization, and single-precision vector addition ($C[i] = A[i] + B[i]$).
   - The Polaris path also supports contiguous row-major FP32 GEMV and GEMM. The matrix kernels use simple dot-product accumulation and have no performance qualification.
   - FP32 Softmax normalizes one vector of finite logits per call with a single wave64 workgroup.
   - FP32 RMSNorm normalizes one vector of finite input values with finite per-element weights and a positive finite epsilon.
   - Other higher-level tensor operations and Transformer LLM layers are outside the current scope.

4. **Multi-GPU Context Partitioning:**
   - The example dispatches to one selected render node per invocation. The validated hardware path is Polaris 10 FP32 vector addition, GEMV, GEMM, Softmax and RMSNorm on the RX 570 and RX 580. See the [hardware validation report](FINAL_VALIDATION_REPORT.md).

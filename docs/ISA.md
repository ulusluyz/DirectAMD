# AMD GPU ISA & Machine Code Research

## Overview
DirectAMD bypasses intermediate compilers at runtime by emitting AMD GPU machine code directly in C++. This document outlines the AMD GPU Instruction Set Architecture (ISA) encoding structures, register layouts, and compute program launching semantics.

---

## 1. Register Architecture

AMD GPUs feature two primary register files per SIMD unit:

### Scalar General Purpose Registers (SGPRs)
* 32-bit registers shared across all threads in a wavefront.
* Used for uniform values, base memory addresses (64-bit pointers occupy two adjacent SGPRs), execution masks (`EXEC`), and wave control state.
* Addressable from `s0` up to `s103`.

### Vector General Purpose Registers (VGPRs)
* 32-bit registers unique to each lane (work-item) within a wavefront.
* Used for thread-divergent data, vector operands, thread IDs, and data buffers.
* Addressable from `v0` up to `v255`.

### Local Data Share (LDS)
* On-chip low-latency SRAM shared among work-items within a workgroup (up to 64 KB per Compute Unit).

---

## 2. Target Instruction Encodings (RDNA2 / RDNA3)

AMD ISA uses 32-bit (single word) and 64-bit (double word) encoding formats.

### SOP2 (Scalar Operations, 2 Inputs, 1 Output) - 32-bit
Format: `[31:30]=10` | `[29:23]=OP` | `[22:16]=SDST` | `[15:8]=SSRC1` | `[7:0]=SSRC0`

Example: `S_ADD_U32 s0, s1, s2`

### VOP2 (Vector Operations, 2 Inputs, 1 Output) - 32-bit
Format: `[31]=0` | `[30:25]=OP` | `[24:17]=VDST` | `[16:9]=VSRC1` | `[8:0]=SRC0`

Example: `V_ADD_F32 v0, v1, v2`

### VOP3 (Vector Operations, 3 Inputs / Extended) - 64-bit
Format Word 0: `[31:26]=110100` | `[25:17]=OP` | `[16:8]=Reserved` | `[7:0]=VDST`
Format Word 1: `[31:23]=SRC2` | `[22:14]=SRC1` | `[13:0]=SRC0`

Example: `V_ADD3_U32 v0, v1, v2, v3` / `V_MAD_F32 v0, v1, v2, v3`

### SOPP (Scalar Control / Program Flow) - 32-bit
Format: `[31:23]=101111111` | `[22:16]=OP` | `[15:0]=SIMM16`

Example: `S_ENDPGM` (`0xBF810000`) terminating shader wave execution.

---

## 3. Minimal Vector-Add Compute Kernel (FP32)

A simple vector-add kernel ($C[i] = A[i] + B[i]$) in raw RDNA ISA contains:

1. **Load base addresses** from user SGPRs into scalar register pairs (`s[0:1]` for A, `s[2:3]` for B, `s[4:5]` for C).
2. **Calculate thread global offset**:
   - Get workgroup ID (`s6`) and local thread ID (`v0`).
   - `v_add_u32 v1, s6, v0` (global element index $i$).
3. **Load operands from VRAM to VGPR**:
   - `GLOBAL_LOAD_DWORD v2, v1, s[0:1]` (Load $A[i]$)
   - `GLOBAL_LOAD_DWORD v3, v1, s[2:3]` (Load $B[i]$)
   - `S_WAITCNT vmcnt(0)` (Wait for vector memory reads)
4. **Vector Addition**:
   - `V_ADD_F32 v4, v2, v3` ($C[i] = A[i] + B[i]$)
5. **Store result to VRAM**:
   - `GLOBAL_STORE_DWORD v1, v4, s[4:5]` (Store $C[i]$)
6. **Program Termination**:
   - `S_ENDPGM` (`0xBF810000`)

---

## 4. Architecture Justification Summary

* **RDNA2 (GFX1030) / RDNA3 (GFX1100)**: Clean VOP2/VOP3 encoding scheme, standard 32-bit Wavefront dispatch, well-documented PM4 headers (`COMPUTE_PGM_LO/HI`, `USER_SGPR_MSB`).
* DirectAMD implements `src/isa/rdna/rdna_encoder.cpp` as an internal, header-backed C++ instruction synthesizer producing machine code vectors without spawning sub-processes or invoking LLVM/ROCm.

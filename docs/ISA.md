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

## 3. RDNA Vector-Add Research Sketch (FP32)

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

## 4. Polaris Vector-Add Kernel (gfx803)

`PolarisEncoder::emit_vector_add_kernel` provides the wave64 code exercised
by the hardware example. User SGPR pairs `s[0:1]`, `s[2:3]`, and
`s[4:5]` hold the A, B, and C addresses. `s6` holds the element count,
`s7` receives workgroup X, and `v0` receives local X. The kernel computes
the index as `workgroup_x * 64 + local_x` and masks lanes at or beyond
the element count. It forms 64-bit addresses with explicit carry,
performs GFX8 flat loads, adds with `v_add_f32`, and stores the result
with a flat store. Wait instructions order memory operations before
arithmetic and wave termination.

## 5. Polaris FP32 Softmax (gfx803)

The kernel in `src/isa/polaris_softmax.cpp` uses lane X and a stride of
64 to index the input. An EXEC mask protects the tail. XOR partner
permutations with `ds_bpermute_b32` combine the wave's partial maxima
and compensated partial sums. For logits `x` and maximum `m`, the
kernel evaluates the exponential as `exp2((x - m) * log2(e))`.

The output pass scales the base-2 exponent by shifting it by 64 when
it is below −64. It compensates by multiplying by
`2^-64 / sum` in place of `1 / sum`. This keeps the exponentiation
itself in the normal range for probabilities representable as FP32
subnormals.

## 6. Polaris FP32 RMSNorm (gfx803)

The kernel in `src/isa/polaris_rmsnorm.cpp` masks accesses beyond the
end of the vector and reduces across the wave with `ds_bpermute_b32`.
For a scale `2^e` based on the largest input magnitude and epsilon,
it accumulates `mean((x * 2^-e)^2)` with compensation and adds
`epsilon * 2^(-2e)` before taking the reciprocal square root. The
output pass combines input and weight mantissas and applies their
combined exponent with the inverse scaling in a single step. This
limits intermediate overflow and underflow.

## 7. Polaris FP32 RoPE (gfx803)

The kernel in `src/isa/polaris_rope.cpp` pairs lanes' inputs as
`(2i, 2i+1)` for interleaved or `(i, i + rotary_dim/2)` for half-split.
It takes a table of FP32 phases in turns, evaluates `v_sin_f32` and
`v_cos_f32`, and applies the two-component rotation in FP32. EXEC masks
route other valid lanes to a bitwise copy of the unrotated suffix and
disable lanes beyond the output.

## 8. Architecture Justification Summary

* **RDNA2 (GFX1030) / RDNA3 (GFX1100)**: Clean VOP2/VOP3 encoding scheme, standard 32-bit Wavefront dispatch, well-documented PM4 headers (`COMPUTE_PGM_LO/HI`, `USER_SGPR_MSB`).
* DirectAMD implements `src/isa/rdna_encoder.cpp` as an internal, header-backed C++ instruction synthesizer producing machine code vectors without spawning sub-processes or invoking LLVM/ROCm.

# AMDGPU Command Submission & PM4 Packet Protocol

## Overview
DirectAMD interacts directly with AMD GPU Command Processors (CP) by encoding PM4 (Packet Type 4 / Type 3) command buffers and submitting them via the Linux AMDGPU CS ioctl.

---

## 1. PM4 Packet Format

AMD CP command streams consist of dword (32-bit) packets.

### PM4 Type 3 Header Structure
```
31         30 29             16 15              8 7               0
┌────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Header Type│  Count (dwords) │     IT_OPCODE   │  Control bits   │
│   (2 bits) │    (14 bits)    │     (8 bits)    │    (8 bits)     │
└────────────┴─────────────────┴─────────────────┴─────────────────┘
```
- **Type 3 Header:** `(3U << 30) | ((count & 0x3FFF) << 16) | ((opcode & 0xFF) << 8)`

---

## 2. Key PM4 Opcodes for Compute Dispatch

| Packet Opcode | Value | Description |
| :--- | :--- | :--- |
| `PACKET3_SET_SH_REG` | `0x76` | Sets Shader registers (e.g., kernel base address, user SGPRs, LDS size). |
| `PACKET3_DISPATCH_DIRECT` | `0x15` | Triggers direct grid execution with specified Workgroup (X, Y, Z) counts. |
| `PACKET3_INDIRECT_BUFFER` | `0x32` | Executes nested indirect command buffers in GPU memory. |
| `PACKET3_ACQUIRE_MEM` | `0x58` | Memory synchronization & L1/L2 cache invalidation/flush packet. |
| `PACKET3_EVENT_WRITE` | `0x46` | Emits GPU events (e.g. Cache flush, EOP fence write). |

---

## 3. Polaris Compute Register Configuration

The builder takes shader register addresses in dwords. `SET_SH_REG`
stores the address relative to the shader register base, `0x2C00`.
The header count is the number of payload dwords minus one.

The dispatch sets the program address with `COMPUTE_PGM_LO/HI` from the
256-byte-aligned GPU address shifted right by eight bits. It configures
64 threads in X and the register allocation in `COMPUTE_PGM_RSRC1/2`.
Vector add, GEMV and GEMM use the first six user SGPRs for three 64-bit buffer
addresses. Vector add then receives the element count, GEMV receives
rows and columns, and GEMM receives `M, N, K`. Workgroup X follows
the user SGPRs, and GEMM also enables workgroup Y.

`DISPATCH_DIRECT` carries four dwords and sets the compute shader type
bit. Vector add and GEMV round the element or row count up to a wave.
GEMM dispatches `ceil(N/64)` workgroups in X and `M` in Y. Single-dword
NOP packets align the indirect buffer to a multiple of eight dwords.

Softmax uses five user SGPRs for two 64-bit buffer addresses and the
vector length. Its PM4 dispatch launches a single wave64 workgroup and
enables FP32 subnormal input and output handling.

RMSNorm uses eight user SGPRs for the input, weight and output
addresses, vector length, and FP32 epsilon bits. Its PM4 dispatch
launches one wave64 workgroup and enables FP32 subnormal handling.

RoPE uses nine user SGPRs for the input, phase table and output
addresses, vector length, rotary dimension, and layout. Its grid
contains one logical work item per rotary pair and one per element
in the unrotated suffix, rounded up to wave64 workgroups.

## 4. Submission via `DRM_IOCTL_AMDGPU_CS`

DirectAMD creates a BO list with `DRM_IOCTL_AMDGPU_BO_LIST` containing
the indirect buffer, kernel code, and A, B and C buffers. The
`AMDGPU_CHUNK_ID_IB` payload specifies the indirect buffer GPU virtual
address, the actual PM4 byte count including NOP padding, and the
compute IP and ring. The unused bytes in the GEM allocation stay
outside the submitted IB.

The CS request's `chunks` field points to an array of 64-bit userspace
addresses, with one address per `drm_amdgpu_cs_chunk`. Passing the chunk
struct address directly causes the driver to interpret its header as
an address and return `EFAULT`. After submission, the userspace BO-list
handle is released. The resulting 64-bit fence sequence is passed to
`DRM_IOCTL_AMDGPU_WAIT_CS` with an absolute `CLOCK_MONOTONIC` deadline
before reading the output.

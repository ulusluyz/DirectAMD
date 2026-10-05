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
│ Header Type│  Count (dwords) │     IT_OPCODE   │    Predicate    │
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

## 3. Register Configuration Packet Sequence

Prior to dispatching a compute wavefront grid, key registers are configured using `PACKET3_SET_SH_REG`:

1. **Kernel Program Base Address:**
   - `COMPUTE_PGM_LO` (`0x2E0C`): Lower 32 bits of GPU Virtual Address divided by 256.
   - `COMPUTE_PGM_HI` (`0x2E0D`): Upper 32 bits of GPU Virtual Address.
2. **Resource SGPR Inputs (User SGPRs):**
   - `COMPUTE_USER_DATA_0` .. `COMPUTE_USER_DATA_15` (`0x2E40`+): Base addresses of memory buffers (A, B, C) and scalar parameters.
3. **Workgroup Dimensions:**
   - `COMPUTE_PGM_RSRC1` / `COMPUTE_PGM_RSRC2`: Register counts (VGPR/SGPR reservation), LDS size per workgroup.
   - `COMPUTE_NUM_THREAD_X/Y/Z`: Threads per workgroup block (e.g., 64 x 1 x 1).

---

## 4. Submission via `DRM_IOCTL_AMDGPU_CS`

Command submission chunks (`struct drm_amdgpu_cs_chunk`) are populated:

1. **IB Chunk (`AMDGPU_CHUNK_ID_IB`):** Contains pointer to command buffer BO, offset, and payload size.
2. **BO List Chunk (`AMDGPU_CHUNK_ID_BO_HANDLES`):** List of GEM handles (kernel code BO, buffer A, buffer B, buffer C) required for memory validation by the Linux kernel scheduler (`amdgpu_cs`).

The kernel driver submits the indirect buffer to the hardware ring (`AMDGPU_HW_IP_COMPUTE`), returning a 64-bit sequence fence number used to await execution completion via `DRM_IOCTL_AMDGPU_WAIT_CS`.

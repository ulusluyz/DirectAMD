# DirectAMD Polaris Validation Report

Validated on 2026-11-16, 22:24 TRT, on Debian 13.7 x86-64 with Linux
`6.12.115+deb13.1-amd64` and the kernel `amdgpu` driver. Built with
GCC 14.2.0, CMake 3.31.6 and Ninja 1.12.1 in Debug mode with
`ENABLE_WERROR=ON`. Debug keeps the CTest assertions enabled.

## CPU and host tests

All five registered CTest tests ran and passed. No tests were skipped.

| Test | Result | Coverage |
| --- | --- | --- |
| `test_device` | Pass | Found two DRM render nodes and opened an AMDGPU node |
| `test_memory` | Pass | Virtual address allocation and alignment |
| `test_isa` | Pass | Host-side instruction encoding and termination |
| `test_command` | Pass | PM4 headers, Polaris dispatch fields and IB padding |
| `test_vector_add` | Pass | CPU reference output `[6, 8, 10, 12]` |

## Physical GPU tests

Each row represents three independent invocations of `vector_add_example`
with the indicated render node. Both cards have PCI ID `1002:67df` and
use the Polaris / GCN4 (`gfx803`) wave64 path.

| Card | PCI address | Render node | Runs passed | GPU submissions completed |
| --- | --- | --- | --- | --- |
| Radeon RX 580, 36 CUs | `0000:0a:00.0` | `/dev/dri/renderD128` | 3/3 | 6/6 |
| Radeon RX 570, 32 CUs | `0000:0b:00.0` | `/dev/dri/renderD129` | 3/3 | 6/6 |

In every run, GPU readback produced `[6, 8, 10, 12]` for the four-element
FP32 case and matched the CPU reference at all 257 positions in the
five-workgroup case. The output sentinels remained intact in all 60 and
63 padded positions, respectively. All six invocations exited with
status 0. Device-filtered `strace` recorded 12 successful
`DRM_IOCTL_AMDGPU_CS` calls and 12 successful `DRM_IOCTL_AMDGPU_WAIT_CS`
calls. An explicit nonexistent render-node path returned exit status 1.

The hardware runs use the Linux AMDGPU DRM ioctls and the C++ runtime.
They exercise device discovery, context creation, GEM buffers, GPU
virtual addressing, compute submission, fences, and GPU result readback.
The scope is the Polaris FP32 vector-add example on these two cards.
RDNA and Vega hardware execution were outside this run.

## Reproduction

Follow the [hardware validation procedure](HARDWARE_VALIDATION.md) to
build, run CTest, and select each GPU by its PCI render-node path.

The earlier Milestone 1 environment (Linux 6.8.0, GCC 13.3.0 and
CMake 3.28.3) passed the build and host tests. It had no DRM render nodes,
so command submission and fence synchronization were skipped in that
environment.

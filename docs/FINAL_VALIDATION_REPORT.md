# DirectAMD Polaris Validation Report

Revalidated on 2026-11-16 at 23:23 TRT on Debian 13.7 x86-64 with Linux
`6.12.115+deb13.1-amd64` and the kernel `amdgpu` driver. The local build
used GCC 14.2.0, CMake 3.31.6 and Ninja 1.12.1 in Debug mode with
`ENABLE_WERROR=ON`, keeping the CTest assertions enabled.

## CPU and host tests

All five registered CTest tests ran and passed. No tests were skipped.

| Test | Result | Coverage |
| --- | --- | --- |
| `test_device` | Pass | Found two DRM render nodes and opened an AMDGPU node |
| `test_memory` | Pass | Virtual address allocation and alignment |
| `test_isa` | Pass | Host-side instruction encoding and termination |
| `test_command` | Pass | PM4 headers, Polaris dispatch fields, shape and length bounds, and IB alignment |
| `test_vector_add` | Pass | CPU reference output `[6, 8, 10, 12]` |

## Physical GPU tests

Each example was invoked three independent times per selected render node.
Both cards have PCI ID `1002:67df` and use Polaris / GCN4 (`gfx803`)
wave64 kernels. Counts below are completed GPU cases compared against CPU
references. Each case uses one command submission and one fence wait.

| Hardware | PCI address | Render node | Vector add (2 cases/run) | GEMV (9 cases/run) | GEMM (10 cases/run) | Softmax (16 cases/run) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Radeon RX 580, 36 CUs | `0000:0a:00.0` | `/dev/dri/renderD128` | 6/6 | 27/27 | 30/30 | 48/48 |
| Radeon RX 570, 32 CUs | `0000:0b:00.0` | `/dev/dri/renderD129` | 6/6 | 27/27 | 30/30 | 48/48 |

All 24 invocations exited with status 0. Device-filtered `strace`
recorded 222 successful `DRM_IOCTL_AMDGPU_CS` calls and 222 successful
`DRM_IOCTL_AMDGPU_WAIT_CS` calls, 111 of each per GPU. Each example
returned exit status 1 for an explicitly nonexistent render node.

### Numerical and buffer checks

| Workload | Sizes covered | CPU comparison |
| --- | --- | --- |
| Vector add | 4 and 257 elements | Exact in both cases, including the four-element result `[6, 8, 10, 12]` |
| GEMV, rows × columns | `1×1`, `3×5`, `63×17`, `64×64`, `65×129`, `127×1`, `257×73`, `1×4097`, `513×1025` | Exact in eight cases. Maximum absolute error `1.32076e-6` for fractional `65×129` |
| GEMM, M × N × K | `1×1×1`, `1×129×5`, `3×5×7`, `7×1×257`, `31×63×65`, `64×64×64`, `65×67×17`, `17×129×33`, `129×65×97`, `67×130×259` | Exact in nine cases. Maximum absolute error `2.06223e-7` for fractional `17×129×33` |
| Softmax, vector length | `1`, `2`, `3`, `7`, `31`, `63`, `64`, `65`, `127`, `257`, `1024`, `4097`, `65537` | Maximum absolute error `7.19476e-8`. Maximum absolute probability-sum deviation from one `1.04308e-7` |

The matrix examples compare every output to a double-precision CPU
reference. The fractional cases use the elementwise bound
`1e-5 + 2e-5 * abs(reference)`. The vector-add output padding and the
GEMV/GEMM GPU and host output sentinels stayed intact. The GEMV example
rejected a short matrix and GEMM rejected short A, B and C spans. The
successful submission counts account only for valid cases.

Each Softmax invocation compared 16 GPU output vectors against an independent
double-precision CPU reference. Cases covered fully negative logits,
values at +10,000, both FP32 finite extremes, subnormal probabilities,
partial waves, a maximum in the final stride, and small terms following
large ones. The probability-sum tolerance was `2e-5` and the elementwise
tolerance was `2e-5 * reference + 2 * FP32_DENORM_MIN`. GPU and host output
sentinels stayed intact. Empty, mismatched, NaN, and infinite inputs were
rejected before submission.

## Scope and reproduction

These results cover direct Linux AMDGPU DRM execution on the listed
cards. They exercise device discovery, GEM allocation and mapping,
compute submission, fences and readback for vector add, GEMV, GEMM and Softmax.
RDNA and Vega hardware execution and other operations are outside this
validation.

Follow the [hardware validation procedure](HARDWARE_VALIDATION.md) to
reproduce the build, CTest run and per-card tests. The earlier Milestone
1 environment had no DRM render nodes and exercised the build and host
tests.

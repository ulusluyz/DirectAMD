# DirectAMD

Experimental low-level LLM inference runtime for AMD GPUs using direct AMDGPU/DRM command submission — without CUDA, ROCm, HIP, Vulkan Compute or OpenCL.

---

## Overview
DirectAMD is an independent, bare-metal C++20 compute runtime that communicates directly with the Linux `amdgpu` kernel driver via DRM render nodes (`/dev/dri/renderD*`). By bypassing high-level vendor stacks (CUDA, ROCm, HIP, Vulkan, OpenCL), DirectAMD gives full direct control over GPU Virtual Address space, PM4 packet dispatch, and instruction-level wave execution.

---

## Goal
The primary objective of DirectAMD is to research and implement low-level GPU compute execution for deep learning operations, targeting direct memory management, custom ISA kernel generation, and command packet submission directly on AMD hardware.

---

## Why DirectAMD?
Existing deep learning runtimes rely on heavy abstraction layers (ROCm, HIP, Vulkan) that introduce memory allocation overhead, indirect runtime launch latency, and complex dependencies. DirectAMD explores the minimal necessary userspace footprint to submit compute work to AMD hardware.

---

## Architecture
```
Host Application
       │
       ▼
DirectAMD Runtime (C++20)
       │
       ├── Device Subsystem (DRM Render Node Query)
       ├── Memory Subsystem (GEM BO VRAM/GTT Allocation & VA Mapping)
       ├── Math Subsystem (Guarded FP32 GEMV and GEMM)
       ├── ISA Subsystem (Zero-Dependency C++ Machine Code Generator)
       └── Command Subsystem (PM4 Packet Builder & CS Submission)
       │
       ▼ ioctl(DRM_IOCTL_AMDGPU_CS)
Linux Kernel amdgpu Driver
       │
       ▼
AMD GPU Hardware (Compute Ring / CUs)
```

---

## Current Status
* **Phase 1 & Phase 2 Complete:**
  - Technical Feasibility & DRM UAPI Kernel Interface Specification complete.
  - Zero-dependency C++ ISA instruction synthesizer for RDNA2/RDNA3 & GCN targets implemented.
  - Device Node Scanner & DRM query interfaces implemented.
  - GEM Buffer Object VRAM/GTT Manager & 48-bit Virtual Address Manager implemented.
  - PM4 Compute Dispatch Command Packet Builder implemented.
  - CPU Reference Vector-Add and Test Suite validated.
* **Hardware execution:** Direct DRM FP32 vector addition, GEMV and GEMM passed on the Radeon RX 570 and RX 580 (Polaris 10 / GCN4 / `gfx803`). All five CTest host tests passed in the same build. See the [validation report](docs/FINAL_VALIDATION_REPORT.md).

---

## Supported Hardware
* **Polaris 10 (GCN4 / gfx803):** Direct DRM vector-add, GEMV and GEMM examples validated on RX 570 and RX 580.
* **RDNA2 (GFX1030 / Navi 21, 22, 23):** Primary initial ISA research target.
* **RDNA3 (GFX1100 / Navi 31, 32, 33):** ISA research target.
* **GCN5 / Vega:** Experimental.

---

## Requirements
* **Operating System:** Linux (Kernel >= 5.4 with `amdgpu` driver)
* **Compiler:** C++20 compliant compiler (GCC >= 11 or Clang >= 13)
* **Build System:** CMake >= 3.20
* **Headers:** DRM kernel headers (`<drm/amdgpu_drm.h>`)

---

## Build Instructions
```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_WERROR=ON ..

# Build binary targets
cmake --build . -j$(nproc)

# Run test suite
ctest --output-on-failure
```

---

## Running Examples
Run the vector-add example from the repository root after building into `build/`.
An optional render-node path selects a specific GPU.

```bash
./build/examples/vector_add_example /dev/dri/renderD128
```

The example checks the readback for 4-element and 257-element vectors and verifies
that lanes beyond the end of each vector leave the output guard values intact.
An unavailable device or a failed dispatch or readback returns a nonzero exit
status. Run `ctest --test-dir build --output-on-failure` for the host-side tests.

### FP32 GEMV on Polaris 10

The `gemv_fp32` runtime API computes `y = A * x` for a contiguous
row-major FP32 matrix. The lengths of `y` and `x` give the row and column
counts. Both dimensions must be positive and the matrix may contain at
most 2^30 elements, subject to available memory. The call waits for the
GPU fence and checks output guards before returning the result.

Run the example with the render node of the card you want to test:

```bash
./build/examples/gemv_example /dev/dri/renderD128
```

The example exercises nine matrix sizes and compares each GPU result
against a double-precision CPU reference.

### FP32 GEMM on Polaris 10

The `gemm_fp32` runtime API computes `C = A * B` for contiguous
row-major FP32 matrices. Pass `M, N, K` and spans for `A[M × K]`,
`B[K × N]`, and `C[M × N]`. Each dimension must be positive and
each matrix may contain at most 2^30 elements, subject to available
memory. The call waits for the GPU fence and checks the output guards
before copying the result to the caller's span.

```bash
./build/examples/gemm_example /dev/dri/renderD128
```

The example compares every GPU result with a double-precision CPU
reference for ten shapes, including rectangular matrices with partial
64-lane workgroups.

---

## Documentation
Comprehensive technical documentation is located in `docs/`:
* [FEASIBILITY_REPORT.md](docs/FEASIBILITY_REPORT.md)
* [AMDGPU_RESEARCH.md](docs/AMDGPU_RESEARCH.md)
* [ISA.md](docs/ISA.md)
* [ARCHITECTURE.md](docs/ARCHITECTURE.md)
* [COMMAND_SUBMISSION.md](docs/COMMAND_SUBMISSION.md)
* [MEMORY.md](docs/MEMORY.md)
* [DEPENDENCY_AUDIT.md](docs/DEPENDENCY_AUDIT.md)
* [LIMITATIONS.md](docs/LIMITATIONS.md)
* [ROADMAP.md](docs/ROADMAP.md)
* [TESTING.md](docs/TESTING.md)
* [BENCHMARKING.md](docs/BENCHMARKING.md)
* [HARDWARE_VALIDATION.md](docs/HARDWARE_VALIDATION.md)
* [FINAL_VALIDATION_REPORT.md](docs/FINAL_VALIDATION_REPORT.md)

---

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution rules and pull request guidelines.

---

## Security
See [SECURITY.md](SECURITY.md) for security policy and vulnerability reporting.

---

## License
DirectAMD is released under the [MIT License](LICENSE).

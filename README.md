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
* **Hardware Execution Status:** `MILESTONE 1: NOT TESTED — PHYSICAL AMD GPU REQUIRED` (Sandbox lacks `/dev/dri/renderD*` device node).

---

## Supported Hardware
* **RDNA2 (GFX1030 / Navi 21, 22, 23):** Primary initial ISA research target.
* **RDNA3 (GFX1100 / Navi 31, 32, 33):** Supported via instruction encoding pipeline.
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
cmake ..

# Build binary targets
cmake --build . -j$(nproc)

# Run test suite
ctest --output-on-failure
```

---

## Running Examples
To run the vector-add example:
```bash
./examples/vector_add/vector_add_example
```
*If physical AMD GPU device node `/dev/dri/renderD128` is present, it will attempt direct GPU command submission. Otherwise, it safely logs `NOT TESTED — PHYSICAL AMD GPU REQUIRED` and runs CPU validation.*

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

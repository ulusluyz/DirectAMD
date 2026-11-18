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
       ├── Math Subsystem (FP32 vector multiply, GEMV, GEMM, Softmax, RMSNorm, RoPE and SiLU)
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
* **Hardware execution:** Direct DRM FP32 vector addition and multiplication, GEMV, GEMM, Softmax, RMSNorm, RoPE and SiLU passed on the Radeon RX 570 and RX 580 (Polaris 10 / GCN4 / `gfx803`). The current CTest suite has seven passing targets. See the [validation report](docs/FINAL_VALIDATION_REPORT.md).

---

## Supported Hardware
* **Polaris 10 (GCN4 / gfx803):** Direct DRM vector-add, vector-mul, GEMV, GEMM, Softmax, RMSNorm, RoPE and SiLU examples validated on RX 570 and RX 580.
* **RDNA2 (GFX1030 / Navi 21, 22, 23):** Primary initial ISA research target.
* **RDNA3 (GFX1100 / Navi 31, 32, 33):** ISA research target.
* **GCN5 / Vega:** Experimental.

---

## Requirements
* **Operating System:** Linux (Kernel >= 5.4 with `amdgpu` driver)
* **Compiler:** C++20 compliant compiler (GCC >= 11 or Clang >= 13)
* **Build System:** CMake >= 3.20
* **Headers:** DRM kernel headers (`<drm/amdgpu_drm.h>`)
* **LLM/converter frontend:** Python >= 3.9 and the PCRE2 8-bit runtime (`libpcre2-8-0` on Debian)

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

### FP32 Softmax on Polaris 10

The `softmax_fp32` API normalizes one nonempty vector of finite FP32
logits. Input and output spans must have the same length. A wave64 GPU
kernel subtracts the maximum before exponentiation, accumulates with
compensation, and scales the small tail to retain subnormal results.
The call waits for the GPU fence and verifies output bounds before
returning the probabilities.

```bash
./build/examples/softmax_example /dev/dri/renderD128
```

The example compares GPU probabilities with a double-precision CPU
reference for lengths up to 65,537, checks the probability sum and
output sentinels, and exercises extreme logits and invalid inputs.

### FP32 RMSNorm on Polaris 10

The `rmsnorm_fp32` API computes
`output[i] = input[i] * weight[i] / sqrt(mean(input^2) + epsilon)`
over one vector of finite FP32 values and weights. The three spans must
have equal nonzero lengths, and epsilon must be positive and finite.
The GPU kernel uses power-of-two scaling and compensated accumulation.
The runtime waits for the GPU fence and checks output guards.

```bash
./build/examples/rmsnorm_example /dev/dri/renderD128
```

The example checks zero, signed, fractional, and extreme inputs and
compares GPU results with a double-precision CPU reference.

### FP32 RoPE on Polaris 10

The `rope_fp32` API takes a vector and a `RopeConfig` with a nonnegative
32-bit position, positive even rotary dimension, finite frequency base
at least 1 (default 10000), and layout. Pair `i` rotates by
`position * base^(-2*i/rotary_dim)` radians. Interleaved pairs consecutive
elements. Half-split pairs the two halves of the rotary prefix.
The GPU copies elements beyond the prefix bit for bit.

The host reduces phases in extended precision before sending them to
the GPU. The `gfx803` kernel computes sine, cosine, and the rotations.
The call waits for the compute fence and checks the output guards.

```bash
./build/examples/rope_example /dev/dri/renderD128
```

The example compares both layouts with an independent double-precision
CPU reference, including partial rotary prefixes and large positions.

### FP32 SiLU on Polaris 10

The `silu_fp32` API computes `x / (1 + exp(-x))` on the GPU for one
nonempty vector of finite FP32 values. Input and output spans must
have the same length and may alias. The kernel evaluates an exponential
of `-abs(x)` to avoid overflow and scales the small negative tail to
preserve subnormal output. The runtime waits for the GPU fence and
checks the output guards before returning.

```bash
./build/examples/silu_example /dev/dri/renderD128
```

The example compares the GPU results with a double-precision CPU
reference and checks in-place calls, signed zeros and FP32 extremes.

### FP32 Elementwise Vector Multiplication on Polaris 10

The `vector_mul_fp32` API computes `C[i] = A[i] * B[i]` on the GPU.
All spans must have the same nonzero length and both inputs must be
finite. The output may alias either input. The PM4 dispatch preserves
FP32 subnormals and rounds to nearest, ties to even. Finite products
outside the FP32 rounding range produce signed infinity.

```bash
./build/examples/vector_mul_example /dev/dri/renderD128
```

The example compares every GPU result bitwise with a double-precision
CPU product rounded to FP32, including signed zeros and subnormal ties.

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


## Qwen2 inference and model conversion

The `directamd-llm` build target connects the FP32 GPU primitives to a
Qwen2 Transformer with persistent VRAM weights and KV cache, a checked
SafeTensors reader, Qwen byte-level BPE, serial prefill and greedy decode.
See [the local inference guide](docs/LLM.md) for the CLI, supported
configuration and validation scope.

`directamd-convert` converts local Qwen2 GGUF F32/F16/Q4_K/Q6_K weights
to FP16 SafeTensors with bounded RAM use and preserves tokenizer/configuration
metadata. See [GGUF conversion](docs/GGUF_CONVERSION.md). Inference converts
weights to FP32 in device-local VRAM using bounded GTT staging.

The local converted Qwen2.5-1.5B model produced the Turkish answer on both
RX 570 and RX 580. See [measured validation](docs/GGUF_VALIDATION.md).
The 64-token upper-bound request ended normally at EOS after 11 tokens.

On Debian, install the build/frontend dependencies with
`sudo apt install build-essential cmake libdrm-dev python3 libpcre2-8-0`.
No Python packages or model downloads are required by the runtime.

```bash
./build/directamd-convert --input /local/model.gguf --output /local/model-fp16
./build/directamd-llm --model /local/model-fp16 \
  --device /dev/dri/renderD128 --prompt "Türkiye'nin başkenti neresidir?" \
  --context-length 160 --max-tokens 64 --memory-mib 7400
```

The conversion output directory must be new. GGUF and SafeTensors model weights
are excluded from Git. See the [pre-publication results](docs/LLM_RELEASE_VALIDATION.md)
for current reruns and the distinction between host, synthetic and trained-model tests.

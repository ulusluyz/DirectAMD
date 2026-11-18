# DirectAMD Physical GPU Validation

The direct DRM vector-add, vector-mul, GEMV, GEMM, Softmax, RMSNorm, RoPE and SiLU examples target Polaris 10
(GCN4 / `gfx803`, PCI ID `1002:67df`). Use a Linux system with the
`amdgpu` driver and read/write access to each GPU's DRM render node.

## Build and run host tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_WERROR=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Debug keeps the host-test assertions enabled.

## Select and test each physical card

```bash
lspci -nnk -d 1002:67df
ls -l /dev/dri/by-path/*-render
```

Use the PCI addresses on your system. The addresses below match the
[tested RX 580 and RX 570](FINAL_VALIDATION_REPORT.md). Each example
selects one card per invocation.

```bash
set -e
for pci in 0000:0a:00.0 0000:0b:00.0; do
    node=$(readlink -f "/dev/dri/by-path/pci-${pci}-render")
    for example in vector_add_example gemv_example gemm_example softmax_example rmsnorm_example rope_example silu_example vector_mul_example; do
        for run in 1 2 3; do
            "./build/examples/$example" "$node"
        done
    done
done
```

To inspect the actual command submissions and fence waits for one run:

```bash
node=$(readlink -f /dev/dri/by-path/pci-0000:0a:00.0-render)
strace -yy -e trace=ioctl -P "$node" ./build/examples/vector_mul_example "$node"
```

A successful invocation reports all its GPU cases and exits 0. Expect
two dispatches for vector add, nine for GEMV, ten for GEMM, or sixteen
each for Softmax and RMSNorm, twenty for RoPE, and fifteen each for
SiLU and vector multiplication.
Each case checks
GPU readback and output boundaries after a completed fence. GEMV and GEMM check rejection of undersized spans.
Softmax checks normalization and rejects empty, mismatched, and
nonfinite inputs. RMSNorm checks its weighted output and rejects
invalid lengths, input values, weights and epsilon. RoPE checks both
layouts against a CPU reference, verifies the unchanged suffix and
rejects invalid spans, rotary dimensions, bases and layouts. SiLU
checks in-place output, signed zeros, extreme values and invalid
inputs. Vector multiplication compares each result bitwise with a
double-precision CPU product rounded to FP32 and checks both output
aliases, signed zeros, subnormal rounding and overflow. A submission, fence or readback failure exits nonzero.
Record GPU execution as untested if there is no usable render node.

## Real Qwen2.5-1.5B inference

The converted local model generated `Türkiye'nin başkenti Ankara'dır.` on
RX 570 and RX 580 with direct AMDGPU/DRM computation. The 64-token upper
bound ended normally at EOS after 11 tokens. See [the detailed model report](GGUF_VALIDATION.md)
and [pre-publication rerun](LLM_RELEASE_VALIDATION.md). These measurements
are separate from the synthetic 64-token fixture and host CTest.

# DirectAMD Testing

## CPU and host tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_WERROR=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

| Test | Coverage |
| --- | --- |
| `test_device` | Render-node enumeration and AMDGPU device queries when a node is available |
| `test_memory` | Host-side GPU virtual-address allocation and alignment |
| `test_isa` | Instruction bit patterns and kernel termination |
| `test_command` | PM4 headers, shader register addressing, matrix shape and vector length bounds, RMSNorm epsilon and RoPE layout encoding, SiLU and vector-mul grid dimensions, subnormal mode and IB alignment |
| `test_vector_add` | CPU reference arithmetic |

## Physical GPU tests

Run the vector-add, vector-mul, GEMV, GEMM, Softmax, RMSNorm, RoPE and SiLU examples separately for each card as described in
[Hardware Validation](HARDWARE_VALIDATION.md). A GPU pass requires
successful DRM submission, a completed fence, and verified GPU readback.
Report CPU results and physical GPU results separately. If hardware is
unavailable, mark GPU execution as untested. See the
[validation report](FINAL_VALIDATION_REPORT.md) for the observed results.

## Qwen2 and converter validation

The current CTest suite has seven targets, including `test_llm_host` and
`test_gguf_converter`. The converter target exercises eight unittest cases.
Run `ctest --test-dir build-llm --output-on-failure` after the build.
Hardware is validated separately from host CTest; a passing host suite does
not establish GPU execution. Run both render nodes' eight example programs,
`directamd-llm --self-test`, and the optional trained-model CPU/GPU tool in
[GGUF_CONVERSION.md](GGUF_CONVERSION.md). Real weights are local-only and are
not included in this repository. See [the release report](LLM_RELEASE_VALIDATION.md)
for the actual executed scope, output and successful fence counts.

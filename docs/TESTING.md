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
| `test_command` | PM4 headers, shader register addressing, matrix shape and vector length bounds, RMSNorm epsilon and RoPE layout encoding, grid dimensions and IB alignment |
| `test_vector_add` | CPU reference arithmetic |

## Physical GPU tests

Run the vector-add, GEMV, GEMM, Softmax, RMSNorm and RoPE examples separately for each card as described in
[Hardware Validation](HARDWARE_VALIDATION.md). A GPU pass requires
successful DRM submission, a completed fence, and verified GPU readback.
Report CPU results and physical GPU results separately. If hardware is
unavailable, mark GPU execution as untested. See the
[validation report](FINAL_VALIDATION_REPORT.md) for the observed results.

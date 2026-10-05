# DirectAMD Testing Policy & Execution Guide

## Policy Architecture

DirectAMD maintains strict testing policies:

1. **Anti-Fake Validation Rule:**
   - Under no circumstances will CPU reference execution or emulation be reported as GPU execution.
   - When running without a physical AMD GPU (`/dev/dri/renderD*`), GPU command submission tests MUST be reported as:
     `SKIPPED — PHYSICAL AMD GPU REQUIRED`
   - Overall hardware milestone status MUST be reported as:
     `MILESTONE 1: NOT TESTED — PHYSICAL AMD GPU REQUIRED`

2. **CPU & Structural Testing:**
   - All host-side software components—device node discovery parser, memory manager bookkeeping, ISA machine code bitfield serialization, PM4 packet generators, and CPU reference math functions—MUST be thoroughly tested and validated on CPU in all build environments.

---

## Running Tests

### Standard Build and Unit Test Suite Execution
```bash
mkdir -p build && cd build
cmake ..
cmake --build .
ctest --output-on-failure
```

### Test Suite Structure
* `tests/test_device.cpp`: Tests DRM render node scanner and architecture matching.
* `tests/test_memory.cpp`: Tests GEM allocation tracking, memory limits, and RAII destruction semantics.
* `tests/test_isa.cpp`: Tests ISA bitfield encoding correctness against known opcode specifications.
* `tests/test_command.cpp`: Tests PM4 packet header and payload construction.
* `tests/test_vector_add.cpp`: Tests CPU math reference and hardware submission checks.

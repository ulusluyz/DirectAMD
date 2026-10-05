# Security Policy

## Security Scope & Low-Level GPU Risk

DirectAMD operates at a low system level, constructing raw PM4 packets and submitting 48-bit Virtual Address descriptors to the Linux `amdgpu` kernel driver. Bugs in command buffer preparation or out-of-bounds GPU virtual addresses can cause GPU hang, kernel TDR (Timeout Detection and Recovery), or GPU reset.

---

## Defensive Measures
1. **Bounds Checking:** All Buffer Object allocation offsets and GPU VA mappings are strictly checked against page boundaries (4 KB alignment).
2. **Deterministic Cleanup:** All GEM BO allocations and GPU virtual address mappings are automatically destroyed via C++ RAII destructors.
3. **No Dynamic Execution of Untrusted Machine Code:** ISA machine code packets are generated using internal structured encoders with explicit opcode validation.

---

## Reporting Vulnerabilities
If you discover a security vulnerability or potential kernel panic / GPU memory corruption exploit in DirectAMD, please report it directly by opening an issue on the repository marked `[SECURITY]`.

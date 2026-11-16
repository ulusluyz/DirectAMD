#include "directamd/isa.hpp"

namespace DirectAMD::ISA {

std::vector<uint32_t> PolarisEncoder::emit_vector_add_kernel() {
    // Assembled with llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx803.
    // s[0:1], s[2:3], s[4:5] are the input/output VAs, s6 is count,
    // s7 is the workgroup X ID, and v0 is the local X ID.
    return {
        // i = workgroup_id_x * 64 + local_id_x. Disable lanes where i >= count.
        0x8e078607, 0x32000007, 0x7d980006, 0x86fe6a7e, 0xbf880013,
        // Load A[i] using a 64-bit address, including the carry from the low dword.
        0x24020082, 0x7e060201, 0x32040200, 0x38060680, 0xdc510000, 0x04000002,
        // Load B[i].
        0x7e060203, 0x32040202, 0x38060680, 0xdc510000, 0x05000002,
        // Wait for the loads and add as FP32.
        0xbf8c0070, 0x02080b04,
        // Store C[i], wait for the store, and end the wave.
        0x7e060205, 0x32040204, 0x38060680, 0xdc710000, 0x00000402, 0xbf8c0070, 0xbf810000,
    };
}

} // namespace DirectAMD::ISA

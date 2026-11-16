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

std::vector<uint32_t> PolarisEncoder::emit_gemv_kernel() {
    // Assembled with llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx803.
    // s[0:1], s[2:3], s[4:5] hold A, x, y. s6 is rows, s7 is columns,
    // s8 is workgroup X, and v0 is local X. Each active lane computes one row.
    return {
        // Convert workgroup and local X to a row index and mask the last wave.
        0x8e088608, 0x32000008, 0x7d980006, 0x86fe6a7e, 0xbf880020,
        // Form A[row][0] and x[0] addresses and initialize the accumulator and column count.
        0xd2850001, 0x00020007, 0x24020282, 0x7e060201, 0x32040200,
        0x38060680, 0x7e080202, 0x7e0a0203, 0x7e100280, 0xbe890007,
        // Load and accumulate one column, then explicitly compare the remaining count to zero.
        0xdc510000, 0x06000002, 0xdc510000, 0x07000004, 0xbf8c0070,
        0x0a0c0f06, 0x02101106, 0x80898109, 0xbf068009, 0xbf850005,
        // Advance both 64-bit addresses by four bytes and loop.
        0x32040484, 0x38060680, 0x32080884, 0x380a0a80, 0xbf82fff1,
        // Store the sum to y[row], wait for the store, and end the wave.
        0x24020082, 0x7e060205, 0x32040204, 0x38060680, 0xdc710000,
        0x00000802, 0xbf8c0070, 0xbf810000,
    };
}

std::vector<uint32_t> PolarisEncoder::emit_gemm_kernel() {
    // Assembled with llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx803.
    // s[0:5] contain the A, B and C addresses. s6/s7/s8 contain M/N/K.
    // s9 is group X, s10 is group Y, and v0 is local X.
    return {
        // Use group Y as the row and mask columns beyond N in the final wave.
        0xbf09060a, 0xbf85002a, 0x8e098609, 0x32000009, 0x7d980007,
        0x86fe6a7e, 0xbf880025,
        // Start A at row*K and B at column. Initialize the FP32 sum and remaining K.
        0x920c080a, 0x8e0c820c, 0x7e04020c, 0x7e060201, 0x32040400,
        0x38060680, 0x24020082, 0x7e0a0203, 0x32080202, 0x380a0a80,
        0x7e100280, 0xbe8b0008, 0x8e0d8207,
        // Load the products' operands, accumulate and stop after exactly K terms.
        0xdc510000, 0x06000002, 0xdc510000, 0x07000004, 0xbf8c0070,
        0x0a0c0f06, 0x02101106, 0x808b810b, 0xbf06800b, 0xbf850005,
        // Advance A by one element and B by N elements using 64-bit addresses.
        0x32040484, 0x38060680, 0x3208080d, 0x380a0a80, 0xbf82fff1,
        // Store C[row*N+column], wait for the store, and terminate the wave.
        0x920c070a, 0x8e0c820c, 0x3202020c, 0x7e060205, 0x32040204,
        0x38060680, 0xdc710000, 0x00000802, 0xbf8c0070, 0xbf810000,
    };
}

} // namespace DirectAMD::ISA

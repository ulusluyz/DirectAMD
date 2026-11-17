#include "directamd/isa.hpp"

namespace DirectAMD::ISA {

std::vector<uint32_t> PolarisEncoder::emit_silu_kernel() {
    // Assembled for gfx803 by llvm-mc. Branch offsets are relative to the next dword.
    // s[0:3] hold input and output addresses, s4 is length, s5 is group X,
    // and v0 is local X.
    return {
        // Mask the partial wave and load the input.
        0x8e058605,               // s_lshl_b32 s5, s5, 6
        0x32000005,               // v_add_u32_e32 v0, vcc, s5, v0
        0x7d980004,               // v_cmp_gt_u32_e32 vcc, s4, v0
        0x86fe6a7e,               // s_and_b64 exec, exec, vcc
        0xbf880026,               // s_cbranch_execz done
        0x24000082,               // v_lshlrev_b32_e32 v0, 2, v0
        0x7e040201,               // v_mov_b32_e32 v2, s1
        0x32020000,               // v_add_u32_e32 v1, vcc, s0, v0
        0x38040480,               // v_addc_u32_e32 v2, vcc, 0, v2, vcc
        0xdc510000, 0x03000001,   // flat_load_dword v3, v[1:2] glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)

        // Form -abs(x) * log2(e). Clamp abs(x) to keep the exponent finite.
        0x260806ff, 0x7fffffff,   // v_and_b32_e32 v4, 0x7fffffff, v3
        0x140808ff, 0x43000000,   // v_min_f32_e32 v4, 0x43000000, v4
        0x0a0808ff, 0xbfb8aa3b,   // v_mul_f32_e32 v4, 0xbfb8aa3b, v4

        // Scale very small exponentials. Their contribution to 1+t rounds away.
        0x7c8808ff, 0xc2800000,   // v_cmp_gt_f32_e32 vcc, 0xc2800000, v4
        0x020a08ff, 0x42800000,   // v_add_f32_e32 v5, 0x42800000, v4
        0x00080b04,               // v_cndmask_b32_e32 v4, v4, v5, vcc
        0x7e084104,               // v_exp_f32_e32 v4, v4
        0x7e0c0280,               // v_mov_b32_e32 v6, 0
        0x000c0d04,               // v_cndmask_b32_e32 v6, v4, v6, vcc

        // Also clamp the negative numerator's input so the far tail rounds to negative zero.
        0x161006ff, 0xc3000000,   // v_max_f32_e32 v8, 0xc3000000, v3
        0x0a0e10ff, 0x1f800000,   // v_mul_f32_e32 v7, 0x1f800000, v8
        0x000e0f08,               // v_cndmask_b32_e32 v7, v8, v7, vcc

        // Use x for nonnegative values and x*t for negative values, then divide by 1+t.
        0x0a100f04,               // v_mul_f32_e32 v8, v4, v7
        0x02120cf2,               // v_add_f32_e32 v9, 1.0, v6
        0x7e124509,               // v_rcp_f32_e32 v9, v9
        0x7c880680,               // v_cmp_gt_f32_e32 vcc, 0, v3
        0x00061103,               // v_cndmask_b32_e32 v3, v3, v8, vcc
        0x0a060709,               // v_mul_f32_e32 v3, v9, v3

        // Store to a guarded output buffer and wait for the write.
        0x7e040203,               // v_mov_b32_e32 v2, s3
        0x32020002,               // v_add_u32_e32 v1, vcc, s2, v0
        0x38040480,               // v_addc_u32_e32 v2, vcc, 0, v2, vcc
        0xdc710000, 0x00000301,   // flat_store_dword v[1:2], v3 glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xbf810000,               // s_endpgm
    };
}

} // namespace DirectAMD::ISA

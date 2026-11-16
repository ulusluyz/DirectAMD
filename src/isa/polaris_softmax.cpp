#include "directamd/isa.hpp"

namespace DirectAMD::ISA {

std::vector<uint32_t> PolarisEncoder::emit_softmax_kernel() {
    // Assembled for gfx803 by llvm-mc. Branch offsets are relative to the next dword.
    // s[0:1] = input, s[2:3] = output, s4 = length, v0 = lane X.
    // s6 is the number of 64-element strides and s[8:9] is the full wave EXEC.
    return {
        // Initialize the max pass. Each lane walks its own strided slice.
        0xbe88017e,               // s_mov_b64 s[8:9], exec
        0x8006bf04,               // s_add_u32 s6, s4, 63
        0x8f068606,               // s_lshr_b32 s6, s6, 6
        0xbe870006,               // s_mov_b32 s7, s6
        0x7e020300,               // v_mov_b32_e32 v1, v0
        0x7e0402ff, 0xff800000,   // v_mov_b32_e32 v2, 0xff800000

        // max_loop: Load only valid indices and retain each lane's running maximum.
        0x7d980204,               // v_cmp_gt_u32_e32 vcc, s4, v1
        0x86fe6a08,               // s_and_b64 exec, s[8:9], vcc
        0x24080282,               // v_lshlrev_b32_e32 v4, 2, v1
        0x7e0c0201,               // v_mov_b32_e32 v6, s1
        0x320a0800,               // v_add_u32_e32 v5, vcc, s0, v4
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc510000, 0x07000005,   // flat_load_dword v7, v[5:6] glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)
        0x16040507,               // v_max_f32_e32 v2, v7, v2
        0xbefe0108,               // s_mov_b64 exec, s[8:9]
        0x320202c0,               // v_add_u32_e32 v1, vcc, 64, v1
        0x80878107,               // s_sub_u32 s7, s7, 1
        0xbf078007,               // s_cmp_lg_u32 s7, 0
        0xbf85fff1,               // s_cbranch_scc1 max_loop
        0xbe8700a0,               // s_mov_b32 s7, 32

        // reduce_max: All lanes participate. XOR partner indices reduce the maximum across the wave.
        0x2a080007,               // v_xor_b32_e32 v4, s7, v0
        0x24080882,               // v_lshlrev_b32_e32 v4, 2, v4
        0xd87e0000, 0x07000204,   // ds_bpermute_b32 v7, v4, v2
        0xbf8c007f,               // s_waitcnt lgkmcnt(0)
        0x16040507,               // v_max_f32_e32 v2, v7, v2
        0x8f078107,               // s_lshr_b32 s7, s7, 1
        0xbf078007,               // s_cmp_lg_u32 s7, 0
        0xbf85fff7,               // s_cbranch_scc1 reduce_max
        0xbe8a00ff, 0x3fb8aa3b,   // s_mov_b32 s10, 0x3fb8aa3b
        0xbe870006,               // s_mov_b32 s7, s6
        0x7e020300,               // v_mov_b32_e32 v1, v0
        0x7e060280,               // v_mov_b32_e32 v3, 0
        0x7e100280,               // v_mov_b32_e32 v8, 0

        // sum_loop: Use a compensated per-lane sum of exp2((x - max) * log2(e)).
        0x7d980204,               // v_cmp_gt_u32_e32 vcc, s4, v1
        0x86fe6a08,               // s_and_b64 exec, s[8:9], vcc
        0x24080282,               // v_lshlrev_b32_e32 v4, 2, v1
        0x7e0c0201,               // v_mov_b32_e32 v6, s1
        0x320a0800,               // v_add_u32_e32 v5, vcc, s0, v4
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc510000, 0x07000005,   // flat_load_dword v7, v[5:6] glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)
        0x040e0507,               // v_sub_f32_e32 v7, v7, v2
        0x0a0e0e0a,               // v_mul_f32_e32 v7, s10, v7
        0x7e0e4107,               // v_exp_f32_e32 v7, v7
        0x040e1107,               // v_sub_f32_e32 v7, v7, v8
        0x02120f03,               // v_add_f32_e32 v9, v3, v7
        0x04100709,               // v_sub_f32_e32 v8, v9, v3
        0x04100f08,               // v_sub_f32_e32 v8, v8, v7
        0x7e060309,               // v_mov_b32_e32 v3, v9
        0xbefe0108,               // s_mov_b64 exec, s[8:9]
        0x320202c0,               // v_add_u32_e32 v1, vcc, 64, v1
        0x80878107,               // s_sub_u32 s7, s7, 1
        0xbf078007,               // s_cmp_lg_u32 s7, 0
        0xbf85ffea,               // s_cbranch_scc1 sum_loop
        0xbe8700a0,               // s_mov_b32 s7, 32

        // reduce_sum: Pairwise sum of the 64 compensated lane sums.
        0x2a080007,               // v_xor_b32_e32 v4, s7, v0
        0x24080882,               // v_lshlrev_b32_e32 v4, 2, v4
        0xd87e0000, 0x07000304,   // ds_bpermute_b32 v7, v4, v3
        0xbf8c007f,               // s_waitcnt lgkmcnt(0)
        0x02060707,               // v_add_f32_e32 v3, v7, v3
        0x8f078107,               // s_lshr_b32 s7, s7, 1
        0xbf078007,               // s_cmp_lg_u32 s7, 0
        0xbf85fff7,               // s_cbranch_scc1 reduce_sum
        0x7e064503,               // v_rcp_f32_e32 v3, v3
        0x0a1006ff, 0x1f800000,   // v_mul_f32_e32 v8, 0x1f800000, v3
        0xbe8b00ff, 0xc2800000,   // s_mov_b32 s11, 0xc2800000
        0xbe8c00ff, 0x42800000,   // s_mov_b32 s12, 0x42800000
        0xbe870006,               // s_mov_b32 s7, s6
        0x7e020300,               // v_mov_b32_e32 v1, v0

        // output_loop: Scale small exponentials by 2^64, then undo the scaling in the normalized product.
        0x7d980204,               // v_cmp_gt_u32_e32 vcc, s4, v1
        0x86fe6a08,               // s_and_b64 exec, s[8:9], vcc
        0x24080282,               // v_lshlrev_b32_e32 v4, 2, v1
        0x7e0c0201,               // v_mov_b32_e32 v6, s1
        0x320a0800,               // v_add_u32_e32 v5, vcc, s0, v4
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc510000, 0x07000005,   // flat_load_dword v7, v[5:6] glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)
        0x040e0507,               // v_sub_f32_e32 v7, v7, v2
        0x0a0e0e0a,               // v_mul_f32_e32 v7, s10, v7
        0x7c880e0b,               // v_cmp_gt_f32_e32 vcc, s11, v7
        0x02120e0c,               // v_add_f32_e32 v9, s12, v7
        0x000e1307,               // v_cndmask_b32_e32 v7, v7, v9, vcc
        0x00121103,               // v_cndmask_b32_e32 v9, v3, v8, vcc
        0x7e0e4107,               // v_exp_f32_e32 v7, v7
        0x0a0e1307,               // v_mul_f32_e32 v7, v7, v9
        0x7e0c0203,               // v_mov_b32_e32 v6, s3
        0x320a0802,               // v_add_u32_e32 v5, vcc, s2, v4
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc710000, 0x00000705,   // flat_store_dword v[5:6], v7 glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xbefe0108,               // s_mov_b64 exec, s[8:9]
        0x320202c0,               // v_add_u32_e32 v1, vcc, 64, v1
        0x80878107,               // s_sub_u32 s7, s7, 1
        0xbf078007,               // s_cmp_lg_u32 s7, 0
        0xbf85ffe4,               // s_cbranch_scc1 output_loop
        0xbf810000,               // s_endpgm
    };
}

} // namespace DirectAMD::ISA

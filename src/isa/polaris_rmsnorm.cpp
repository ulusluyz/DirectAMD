#include "directamd/isa.hpp"

namespace DirectAMD::ISA {

std::vector<uint32_t> PolarisEncoder::emit_rmsnorm_kernel() {
    // Assembled for gfx803 by llvm-mc. Branch offsets are relative to the next dword.
    // s[0:5] hold x, weight and output addresses. s6 is length, s7 is epsilon.
    // v0 is lane X, s9 is the stride count, and s[12:13] save the wave's EXEC.
    return {
        // Initialize the strided maximum-magnitude pass.
        0xbe8c017e,               // s_mov_b64 s[12:13], exec
        0x8009bf06,               // s_add_u32 s9, s6, 63
        0x8f098609,               // s_lshr_b32 s9, s9, 6
        0xbe8a0009,               // s_mov_b32 s10, s9
        0x7e020300,               // v_mov_b32_e32 v1, v0
        0x7e0402ff, 0x19800000,   // v_mov_b32_e32 v2, 0x19800000

        // max_loop: Mask invalid lanes before reading x and accumulate the local maximum magnitude.
        0x7d980206,               // v_cmp_gt_u32_e32 vcc, s6, v1
        0x86fe6a0c,               // s_and_b64 exec, s[12:13], vcc
        0x24080282,               // v_lshlrev_b32_e32 v4, 2, v1
        0x7e0c0201,               // v_mov_b32_e32 v6, s1
        0x320a0800,               // v_add_u32_e32 v5, vcc, s0, v4
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc510000, 0x07000005,   // flat_load_dword v7, v[5:6] glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)
        0x260e0eff, 0x7fffffff,   // v_and_b32_e32 v7, 0x7fffffff, v7
        0x16040507,               // v_max_f32_e32 v2, v7, v2
        0xbefe010c,               // s_mov_b64 exec, s[12:13]
        0x320202c0,               // v_add_u32_e32 v1, vcc, 64, v1
        0x808a810a,               // s_sub_u32 s10, s10, 1
        0xbf07800a,               // s_cmp_lg_u32 s10, 0
        0xbf85ffef,               // s_cbranch_scc1 max_loop
        0xbe8a00a0,               // s_mov_b32 s10, 32

        // reduce_max: Combine lane maxima, then choose a power of two based on |x| and sqrt(epsilon).
        0x2a08000a,               // v_xor_b32_e32 v4, s10, v0
        0x24080882,               // v_lshlrev_b32_e32 v4, 2, v4
        0xd87e0000, 0x07000204,   // ds_bpermute_b32 v7, v4, v2
        0xbf8c007f,               // s_waitcnt lgkmcnt(0)
        0x16040507,               // v_max_f32_e32 v2, v7, v2
        0x8f0a810a,               // s_lshr_b32 s10, s10, 1
        0xbf07800a,               // s_cmp_lg_u32 s10, 0
        0xbf85fff7,               // s_cbranch_scc1 reduce_max
        0x7e046702,               // v_frexp_exp_i32_f32_e32 v2, v2
        0x7e146607,               // v_frexp_exp_i32_f32_e32 v10, s7
        0x32141481,               // v_add_u32_e32 v10, vcc, 1, v10
        0x22141481,               // v_ashrrev_i32_e32 v10, 1, v10
        0x1a04050a,               // v_max_i32_e32 v2, v10, v2
        0x34040480,               // v_sub_u32_e32 v2, vcc, 0, v2
        0x24140481,               // v_lshlrev_b32_e32 v10, 1, v2
        0xd288000a, 0x00021407,   // v_ldexp_f32 v10, s7, v10
        0xbe8a0009,               // s_mov_b32 s10, s9
        0x7e020300,               // v_mov_b32_e32 v1, v0
        0x7e060280,               // v_mov_b32_e32 v3, 0
        0x7e100280,               // v_mov_b32_e32 v8, 0

        // sum_loop: Square scaled inputs and accumulate with compensation.
        0x7d980206,               // v_cmp_gt_u32_e32 vcc, s6, v1
        0x86fe6a0c,               // s_and_b64 exec, s[12:13], vcc
        0x24080282,               // v_lshlrev_b32_e32 v4, 2, v1
        0x7e0c0201,               // v_mov_b32_e32 v6, s1
        0x320a0800,               // v_add_u32_e32 v5, vcc, s0, v4
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc510000, 0x07000005,   // flat_load_dword v7, v[5:6] glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xd2880007, 0x00020507,   // v_ldexp_f32 v7, v7, v2
        0x0a0e0f07,               // v_mul_f32_e32 v7, v7, v7
        0x040e1107,               // v_sub_f32_e32 v7, v7, v8
        0x02120f03,               // v_add_f32_e32 v9, v3, v7
        0x04100709,               // v_sub_f32_e32 v8, v9, v3
        0x04100f08,               // v_sub_f32_e32 v8, v8, v7
        0x7e060309,               // v_mov_b32_e32 v3, v9
        0xbefe010c,               // s_mov_b64 exec, s[12:13]
        0x320202c0,               // v_add_u32_e32 v1, vcc, 64, v1
        0x808a810a,               // s_sub_u32 s10, s10, 1
        0xbf07800a,               // s_cmp_lg_u32 s10, 0
        0xbf85ffea,               // s_cbranch_scc1 sum_loop
        0xbe8a00a0,               // s_mov_b32 s10, 32

        // reduce_sum: Reduce the lane sums, divide by length, add scaled epsilon and take rsqrt.
        0x2a08000a,               // v_xor_b32_e32 v4, s10, v0
        0x24080882,               // v_lshlrev_b32_e32 v4, 2, v4
        0xd87e0000, 0x07000304,   // ds_bpermute_b32 v7, v4, v3
        0xbf8c007f,               // s_waitcnt lgkmcnt(0)
        0x02060707,               // v_add_f32_e32 v3, v7, v3
        0x8f0a810a,               // s_lshr_b32 s10, s10, 1
        0xbf07800a,               // s_cmp_lg_u32 s10, 0
        0xbf85fff7,               // s_cbranch_scc1 reduce_sum
        0x7e120c06,               // v_cvt_f32_u32_e32 v9, s6
        0x7e124509,               // v_rcp_f32_e32 v9, v9
        0x0a060709,               // v_mul_f32_e32 v3, v9, v3
        0x0206070a,               // v_add_f32_e32 v3, v10, v3
        0x7e064903,               // v_rsq_f32_e32 v3, v3
        0xbe8a0009,               // s_mov_b32 s10, s9
        0x7e020300,               // v_mov_b32_e32 v1, v0

        // output_loop: Form the weighted result from mantissas and apply the combined exponent once.
        0x7d980206,               // v_cmp_gt_u32_e32 vcc, s6, v1
        0x86fe6a0c,               // s_and_b64 exec, s[12:13], vcc
        0x24080282,               // v_lshlrev_b32_e32 v4, 2, v1
        0x7e0c0201,               // v_mov_b32_e32 v6, s1
        0x320a0800,               // v_add_u32_e32 v5, vcc, s0, v4
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc510000, 0x07000005,   // flat_load_dword v7, v[5:6] glc
        0x7e0c0203,               // v_mov_b32_e32 v6, s3
        0x320a0802,               // v_add_u32_e32 v5, vcc, s2, v4
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc510000, 0x09000005,   // flat_load_dword v9, v[5:6] glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)
        0x7e146707,               // v_frexp_exp_i32_f32_e32 v10, v7
        0x7e166709,               // v_frexp_exp_i32_f32_e32 v11, v9
        0x7e0e6907,               // v_frexp_mant_f32_e32 v7, v7
        0x7e126909,               // v_frexp_mant_f32_e32 v9, v9
        0x3214170a,               // v_add_u32_e32 v10, vcc, v10, v11
        0x3214050a,               // v_add_u32_e32 v10, vcc, v10, v2
        0x0a0e1307,               // v_mul_f32_e32 v7, v7, v9
        0x0a0e0707,               // v_mul_f32_e32 v7, v7, v3
        0xd2880007, 0x00021507,   // v_ldexp_f32 v7, v7, v10
        0x7e0c0205,               // v_mov_b32_e32 v6, s5
        0x320a0804,               // v_add_u32_e32 v5, vcc, s4, v4
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc710000, 0x00000705,   // flat_store_dword v[5:6], v7 glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xbefe010c,               // s_mov_b64 exec, s[12:13]
        0x320202c0,               // v_add_u32_e32 v1, vcc, 64, v1
        0x808a810a,               // s_sub_u32 s10, s10, 1
        0xbf07800a,               // s_cmp_lg_u32 s10, 0
        0xbf85ffdd,               // s_cbranch_scc1 output_loop
        0xbf810000,               // s_endpgm
    };
}

} // namespace DirectAMD::ISA

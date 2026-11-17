#include "directamd/isa.hpp"

namespace DirectAMD::ISA {

std::vector<uint32_t> PolarisEncoder::emit_rope_kernel() {
    // Assembled for gfx803 by llvm-mc. Branch offsets are relative to the next dword.
    // s[0:5] hold input, phase table and output addresses. s6 is vector length,
    // s7 is rotary dimension, s8 is layout, s9 is group X and v0 is lane X.
    return {
        // Mask invalid work items and select the rotated pairs. Save the valid EXEC mask.
        0x8f0a8107,               // s_lshr_b32 s10, s7, 1
        0x808b0a06,               // s_sub_u32 s11, s6, s10
        0x8e098609,               // s_lshl_b32 s9, s9, 6
        0x32000009,               // v_add_u32_e32 v0, vcc, s9, v0
        0x7d98000b,               // v_cmp_gt_u32_e32 vcc, s11, v0
        0x86fe6a7e,               // s_and_b64 exec, exec, vcc
        0xbf88003e,               // s_cbranch_execz done
        0x7d98000a,               // v_cmp_gt_u32_e32 vcc, s10, v0
        0xbe8c206a,               // s_and_saveexec_b64 s[12:13], vcc
        0xbf88002c,               // s_cbranch_execz tail
        0xbf068008,               // s_cmp_eq_u32 s8, 0
        0xbf840003,               // s_cbranch_scc0 half_split
        0x24020081,               // v_lshlrev_b32_e32 v1, 1, v0
        0x32040281,               // v_add_u32_e32 v2, vcc, 1, v1
        0xbf820002,               // s_branch load_pair

        // half_split: Pair i with i + rotary_dim / 2 for the half-split layout.
        0x7e020300,               // v_mov_b32_e32 v1, v0
        0x3204000a,               // v_add_u32_e32 v2, vcc, s10, v0

        // load_pair: Read both input elements and the range-reduced phase in turns.
        0x24020282,               // v_lshlrev_b32_e32 v1, 2, v1
        0x24040482,               // v_lshlrev_b32_e32 v2, 2, v2
        0x7e0c0201,               // v_mov_b32_e32 v6, s1
        0x320a0200,               // v_add_u32_e32 v5, vcc, s0, v1
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc510000, 0x07000005,   // flat_load_dword v7, v[5:6] glc
        0x7e0c0201,               // v_mov_b32_e32 v6, s1
        0x320a0400,               // v_add_u32_e32 v5, vcc, s0, v2
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc510000, 0x08000005,   // flat_load_dword v8, v[5:6] glc
        0x24080082,               // v_lshlrev_b32_e32 v4, 2, v0
        0x7e0c0203,               // v_mov_b32_e32 v6, s3
        0x320a0802,               // v_add_u32_e32 v5, vcc, s2, v4
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc510000, 0x09000005,   // flat_load_dword v9, v[5:6] glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)

        // Evaluate sine and cosine on the GPU, then rotate the pair.
        0x7e145309,               // v_sin_f32_e32 v10, v9
        0x7e165509,               // v_cos_f32_e32 v11, v9
        0x0a181707,               // v_mul_f32_e32 v12, v7, v11
        0x0a1a1508,               // v_mul_f32_e32 v13, v8, v10
        0x04181b0c,               // v_sub_f32_e32 v12, v12, v13
        0x0a1a1507,               // v_mul_f32_e32 v13, v7, v10
        0x0a1c1708,               // v_mul_f32_e32 v14, v8, v11
        0x021a1d0d,               // v_add_f32_e32 v13, v13, v14
        0x7e0c0205,               // v_mov_b32_e32 v6, s5
        0x320a0204,               // v_add_u32_e32 v5, vcc, s4, v1
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc710000, 0x00000c05,   // flat_store_dword v[5:6], v12 glc
        0x7e0c0205,               // v_mov_b32_e32 v6, s5
        0x320a0404,               // v_add_u32_e32 v5, vcc, s4, v2
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc710000, 0x00000d05,   // flat_store_dword v[5:6], v13 glc

        // tail: Switch EXEC to the remaining valid lanes and copy the unrotated suffix.
        0x88fe0c7e,               // s_xor_b64 exec, exec, s[12:13]
        0xbf88000d,               // s_cbranch_execz done
        0x3202000a,               // v_add_u32_e32 v1, vcc, s10, v0
        0x24020282,               // v_lshlrev_b32_e32 v1, 2, v1
        0x7e0c0201,               // v_mov_b32_e32 v6, s1
        0x320a0200,               // v_add_u32_e32 v5, vcc, s0, v1
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc510000, 0x07000005,   // flat_load_dword v7, v[5:6] glc
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)
        0x7e0c0205,               // v_mov_b32_e32 v6, s5
        0x320a0204,               // v_add_u32_e32 v5, vcc, s4, v1
        0x380c0c80,               // v_addc_u32_e32 v6, vcc, 0, v6, vcc
        0xdc710000, 0x00000705,   // flat_store_dword v[5:6], v7 glc

        // done: Wait for all stores before ending the wave.
        0xbf8c0070,               // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xbf810000,               // s_endpgm
    };
}

} // namespace DirectAMD::ISA

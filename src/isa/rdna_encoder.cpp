#include "directamd/isa.hpp"

namespace DirectAMD {
namespace ISA {

uint32_t RDNAEncoder::encode_sopp(uint16_t op, uint16_t simm16) {
    // SOPP format: [31:23]=101111111, [22:16]=OP, [15:0]=SIMM16
    return (0b101111111U << 23) | ((static_cast<uint32_t>(op) & 0x7F) << 16) | (simm16 & 0xFFFF);
}

uint32_t RDNAEncoder::encode_sop2(uint8_t op, uint8_t sdst, uint8_t ssrc0, uint8_t ssrc1) {
    // SOP2 format: [31:30]=10, [29:23]=OP, [22:16]=SDST, [15:8]=SSRC1, [7:0]=SSRC0
    return (0b10U << 30) |
           ((static_cast<uint32_t>(op) & 0x7F) << 23) |
           ((static_cast<uint32_t>(sdst) & 0x7F) << 16) |
           ((static_cast<uint32_t>(ssrc1) & 0xFF) << 8) |
           (static_cast<uint32_t>(ssrc0) & 0xFF);
}

uint32_t RDNAEncoder::encode_vop2(uint8_t op, uint8_t vdst, uint8_t src0, uint8_t vsrc1) {
    // VOP2 format: [31]=0, [30:25]=OP, [24:17]=VDST, [16:9]=VSRC1, [8:0]=SRC0
    return ((static_cast<uint32_t>(op) & 0x3F) << 25) |
           ((static_cast<uint32_t>(vdst) & 0xFF) << 17) |
           ((static_cast<uint32_t>(vsrc1) & 0xFF) << 9) |
           (static_cast<uint32_t>(src0) & 0x1FF);
}

uint32_t RDNAEncoder::encode_sopp_waitcnt(uint16_t vmcnt, uint16_t lgkmcnt) {
    // S_WAITCNT: OP = 0x0C in SOPP
    // Bit field simm16: [3:0]=vmcnt, [13:8]=lgkmcnt
    uint16_t simm16 = (vmcnt & 0x0F) | ((lgkmcnt & 0x3F) << 8);
    return encode_sopp(0x0C, simm16);
}

void RDNAEncoder::emit_global_load_dword(std::vector<uint32_t>& code, uint8_t vdst, uint8_t vaddr, uint8_t sbase) {
    // GLOBAL_LOAD_DWORD (RDNA 64-bit encoding)
    // Word 0: [31:26]=110111, [25:18]=OP, [17:16]=0, [15:8]=SBASE, [7:0]=VDST
    // Word 1: [31:0]=VADDR & offset
    uint32_t word0 = (0b110111U << 26) | (0x14U << 18) | ((static_cast<uint32_t>(sbase) & 0xFF) << 8) | (vdst & 0xFF);
    uint32_t word1 = (static_cast<uint32_t>(vaddr) & 0xFF);
    code.push_back(word0);
    code.push_back(word1);
}

void RDNAEncoder::emit_global_store_dword(std::vector<uint32_t>& code, uint8_t vaddr, uint8_t vdata, uint8_t sbase) {
    // GLOBAL_STORE_DWORD (RDNA 64-bit encoding)
    // Word 0: [31:26]=110111, [25:18]=OP (0x1C for store_dword), [15:8]=SBASE, [7:0]=VDATA
    // Word 1: [31:0]=VADDR & offset
    uint32_t word0 = (0b110111U << 26) | (0x1CU << 18) | ((static_cast<uint32_t>(sbase) & 0xFF) << 8) | (vdata & 0xFF);
    uint32_t word1 = (static_cast<uint32_t>(vaddr) & 0xFF);
    code.push_back(word0);
    code.push_back(word1);
}

std::vector<uint32_t> RDNAEncoder::emit_vector_add_kernel() {
    std::vector<uint32_t> code;

    // 1. Calculate byte offset = v0 * 4 bytes (v_lshlrev_b32 v1, 2, v0)
    // OP 0x14 = V_LSHLREV_B32
    code.push_back(encode_vop2(0x14, 1, 128 + 2 /* literal inline 2 */, 0 /* v0 */));

    // 2. Load A[i] -> v2 (global_load_dword v2, v1, s[0:1])
    emit_global_load_dword(code, 2, 1, 0);

    // 3. Load B[i] -> v3 (global_load_dword v3, v1, s[2:3])
    emit_global_load_dword(code, 3, 1, 2);

    // 4. Wait for vector memory reads to complete: s_waitcnt vmcnt(0)
    code.push_back(encode_sopp_waitcnt(0, 0));

    // 5. Add FP32: v4 = v2 + v3 (v_add_f32 v4, v2, v3)
    // OP 0x03 = V_ADD_F32
    code.push_back(encode_vop2(0x03, 4, 2, 3));

    // 6. Store C[i] <- v4 (global_store_dword v1, v4, s[4:5])
    emit_global_store_dword(code, 1, 4, 4);

    // 7. Wait for store: s_waitcnt vmcnt(0)
    code.push_back(encode_sopp_waitcnt(0, 0));

    // 8. Terminate wavefront
    code.push_back(SOPP_S_ENDPGM);

    return code;
}

} // namespace ISA
} // namespace DirectAMD

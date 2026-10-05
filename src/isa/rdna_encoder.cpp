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

std::vector<uint32_t> RDNAEncoder::emit_vector_add_kernel() {
    std::vector<uint32_t> code;

    // 1. S_NOP 0
    code.push_back(encode_sopp(0, 0));

    // 2. V_ADD_F32 v0, v0, v1 (Opcode 0x03 on RDNA)
    code.push_back(encode_vop2(0x03, 0, 0, 1));

    // 3. S_NOP 0
    code.push_back(encode_sopp(0, 0));

    // 4. S_ENDPGM
    code.push_back(SOPP_S_ENDPGM);

    return code;
}

} // namespace ISA
} // namespace DirectAMD

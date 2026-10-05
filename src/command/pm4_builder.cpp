#include "directamd/command.hpp"

namespace DirectAMD {

uint32_t PM4Builder::build_type3_header(uint8_t opcode, uint16_t count) {
    // PM4 Type 3 Header: [31:30]=11, [29:16]=count, [15:8]=opcode, [7:0]=predicate
    return (0b11U << 30) | ((static_cast<uint32_t>(count) & 0x3FFF) << 16) | ((static_cast<uint32_t>(opcode) & 0xFF) << 8);
}

void PM4Builder::add_set_sh_reg(uint32_t reg_offset, const std::vector<uint32_t>& values) {
    uint16_t count = static_cast<uint16_t>(values.size());
    packets_.push_back(build_type3_header(0x76, count)); // 0x76 = PACKET3_SET_SH_REG
    packets_.push_back(reg_offset & 0xFFFF);
    for (uint32_t v : values) {
        packets_.push_back(v);
    }
}

void PM4Builder::add_dispatch_direct(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t dispatch_initiator) {
    packets_.push_back(build_type3_header(0x15, 4)); // 0x15 = PACKET3_DISPATCH_DIRECT
    packets_.push_back(gx);
    packets_.push_back(gy);
    packets_.push_back(gz);
    packets_.push_back(dispatch_initiator);
}

} // namespace DirectAMD

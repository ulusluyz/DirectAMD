#ifndef DIRECTAMD_ISA_HPP
#define DIRECTAMD_ISA_HPP

#include "directamd/common.hpp"
#include <cstdint>
#include <vector>

namespace DirectAMD {
namespace ISA {

// RDNA / GCN Opcodes
constexpr uint32_t SOPP_S_ENDPGM = 0xBF810000;
constexpr uint32_t SOPP_S_NOP    = 0xBF800000;

class RDNAEncoder {
public:
    static uint32_t encode_sopp(uint16_t op, uint16_t simm16 = 0);
    static uint32_t encode_sop2(uint8_t op, uint8_t sdst, uint8_t ssrc0, uint8_t ssrc1);
    static uint32_t encode_vop2(uint8_t op, uint8_t vdst, uint8_t src0, uint8_t vsrc1);

    // Synthesizes complete vector add compute kernel in AMD GPU machine instructions
    static std::vector<uint32_t> emit_vector_add_kernel();
};

} // namespace ISA
} // namespace DirectAMD

#endif // DIRECTAMD_ISA_HPP

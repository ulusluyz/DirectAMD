#ifndef DIRECTAMD_ISA_HPP
#define DIRECTAMD_ISA_HPP

#include "directamd/common.hpp"
#include <cstdint>
#include <vector>

namespace DirectAMD {
namespace ISA {

// SOPP Opcodes
constexpr uint32_t SOPP_S_NOP    = 0xBF800000;
constexpr uint32_t SOPP_S_ENDPGM = 0xBF810000;

class RDNAEncoder {
public:
    static uint32_t encode_sopp(uint16_t op, uint16_t simm16 = 0);
    static uint32_t encode_sop2(uint8_t op, uint8_t sdst, uint8_t ssrc0, uint8_t ssrc1);
    static uint32_t encode_vop2(uint8_t op, uint8_t vdst, uint8_t src0, uint8_t vsrc1);
    static uint32_t encode_sobc(uint8_t op, uint16_t simm16);
    static uint32_t encode_sopp_waitcnt(uint16_t vmcnt, uint16_t lgkmcnt);

    // Encodes 64-bit GLOBAL_LOAD_DWORD / GLOBAL_STORE_DWORD instructions for RDNA
    static void emit_global_load_dword(std::vector<uint32_t>& code, uint8_t vdst, uint8_t vaddr, uint8_t sbase);
    static void emit_global_store_dword(std::vector<uint32_t>& code, uint8_t vaddr, uint8_t vdata, uint8_t sbase);

    // Synthesizes complete vector add compute kernel in AMD GPU RDNA machine instructions
    static std::vector<uint32_t> emit_vector_add_kernel();
};

} // namespace ISA
} // namespace DirectAMD

#endif // DIRECTAMD_ISA_HPP

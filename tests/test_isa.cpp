#include "directamd/isa.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[TEST] ISA Encoding Test...\n";

    uint32_t endpgm = DirectAMD::ISA::RDNAEncoder::encode_sopp(1, 0); // S_ENDPGM op 1
    assert((endpgm & 0xFFFF0000) == 0xBF810000);

    auto kernel = DirectAMD::ISA::RDNAEncoder::emit_vector_add_kernel();
    assert(!kernel.empty());
    assert(kernel.back() == DirectAMD::ISA::SOPP_S_ENDPGM);

    std::cout << "[+] Encoded " << kernel.size() << " instructions in vector add kernel binary.\n";
    std::cout << "[PASS] ISA Encoding Test Completed.\n";
    return 0;
}

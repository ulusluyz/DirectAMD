#include "directamd/command.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[TEST] PM4 Packet Builder Test...\n";

    DirectAMD::PM4Builder builder;
    builder.add_set_sh_reg(0x2E0C, {0x1234, 0x5678});
    builder.add_dispatch_direct(1, 1, 1);

    const auto& packets = builder.get_packets();
    assert(packets.size() == 9); // Header(1) + reg(1) + val(2) + Header(1) + gx/gy/gz/init(4)

    // Check Type 3 Header bits [31:30] == 0b11
    assert((packets[0] >> 30) == 0b11);

    assert(packets[0] == 0xC0027600);
    assert(packets[1] == 0x20C);
    assert(packets[4] == 0xC0031502);

    DirectAMD::PM4Builder polaris;
    polaris.add_polaris_vector_add(0x100003000ULL, {
        0x100000000ULL, 0x100001000ULL, 0x100002000ULL, 257
    });
    const auto& dispatch = polaris.get_packets();
    assert(dispatch.size() == 40);
    assert(dispatch[10] == 0xC0027600);
    assert(dispatch[11] == 0x20C);
    assert(dispatch[12] == 0x01000030);
    assert(dispatch[32] == 257);
    assert(dispatch[33] == 0xC0031502);
    assert(dispatch[34] == 5);
    assert(dispatch[38] == 0xFFFF1000);
    assert(dispatch[39] == 0xFFFF1000);

    std::cout << "[+] Serialized " << packets.size() << " dwords of PM4 compute packets.\n";
    std::cout << "[PASS] PM4 Packet Builder Test Completed.\n";
    return 0;
}

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

    std::cout << "[+] Serialized " << packets.size() << " dwords of PM4 compute packets.\n";
    std::cout << "[PASS] PM4 Packet Builder Test Completed.\n";
    return 0;
}

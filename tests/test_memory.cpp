#include "directamd/memory.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[TEST] Memory Virtual Address Allocation Test...\n";

    DirectAMD::VAManager va_mgr(0x100000000ULL);

    uint64_t va1 = va_mgr.allocate_range(1024, 4096);
    uint64_t va2 = va_mgr.allocate_range(8192, 4096);

    assert(va1 == 0x100000000ULL);
    assert(va2 == 0x100001000ULL); // 4096 bytes after va1

    std::cout << "[+] VA1: 0x" << std::hex << va1 << "\n";
    std::cout << "[+] VA2: 0x" << std::hex << va2 << std::dec << "\n";

    std::cout << "[PASS] Memory VA Allocation Test Completed.\n";
    return 0;
}

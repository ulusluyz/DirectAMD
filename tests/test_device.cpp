#include "directamd/device.hpp"
#include "directamd/error.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[TEST] Device Discovery Test...\n";

    auto nodes = DirectAMD::Device::scan_render_nodes();
    std::cout << "[+] Found " << nodes.size() << " DRM render node(s).\n";

    std::error_code ec;
    auto dev = DirectAMD::Device::open_first_available(ec);

    if (!dev) {
        assert(ec == DirectAMD::ErrorCode::DEVICE_NOT_FOUND);
        std::cout << "[+] Correctly reported DEVICE_NOT_FOUND in environment without render nodes.\n";
    } else {
        std::cout << "[+] Opened device at " << dev->get_properties().device_path << "\n";
    }

    std::cout << "[PASS] Device Discovery Test Completed.\n";
    return 0;
}

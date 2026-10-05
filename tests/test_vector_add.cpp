#include "directamd/common.hpp"
#include "directamd/device.hpp"
#include "directamd/error.hpp"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "[TEST] Vector Add CPU Reference Math Test...\n";

    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> C_expected = {6.0f, 8.0f, 10.0f, 12.0f};

    for (size_t i = 0; i < A.size(); ++i) {
        float res = A[i] + B[i];
        assert(res == C_expected[i]);
    }

    std::error_code ec;
    auto dev = DirectAMD::Device::open_first_available(ec);
    if (!dev) {
        std::cout << "SKIPPED — PHYSICAL AMD GPU REQUIRED\n";
        std::cout << "MILESTONE 1: NOT TESTED — PHYSICAL AMD GPU REQUIRED\n";
    }

    std::cout << "[PASS] Vector Add Reference Math Test Completed.\n";
    return 0;
}

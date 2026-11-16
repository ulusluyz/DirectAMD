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

    static_assert(DirectAMD::GemvKernelParams::valid_shape(1, 1u << 30));
    static_assert(!DirectAMD::GemvKernelParams::valid_shape(65536, 65536));
    static_assert(!DirectAMD::GemvKernelParams::valid_shape(0, 5));
    static_assert(!DirectAMD::GemvKernelParams::valid_shape(5, 0));

    DirectAMD::PM4Builder gemv;
    gemv.add_polaris_gemv(0x100003000ULL, {
        0x100000000ULL, 0x100001100ULL, 0x100002100ULL, 65, 129
    });
    const auto& gemv_dispatch = gemv.get_packets();
    assert(gemv_dispatch.size() == 40);
    assert(gemv_dispatch[16] == 0xC0042); // 12 VGPRs, 16 SGPRs, FP mode
    assert(gemv_dispatch[17] == 0x90); // Eight user SGPRs and group ID X
    assert(gemv_dispatch[24] == 0xC0087600);
    assert(gemv_dispatch[28] == 0x1100);
    assert(gemv_dispatch[30] == 0x2100);
    assert(gemv_dispatch[32] == 65);
    assert(gemv_dispatch[33] == 129);
    assert(gemv_dispatch[34] == 0xC0031502);
    assert(gemv_dispatch[35] == 2);
    assert(gemv_dispatch[39] == 0xFFFF1000);

    static_assert(DirectAMD::GemmKernelParams::valid_shape(65, 129, 33));
    static_assert(!DirectAMD::GemmKernelParams::valid_shape(0, 129, 33));
    static_assert(!DirectAMD::GemmKernelParams::valid_shape(65, 0, 33));
    static_assert(!DirectAMD::GemmKernelParams::valid_shape(65, 129, 0));
    static_assert(!DirectAMD::GemmKernelParams::valid_shape(65536, 1, 65536));
    static_assert(!DirectAMD::GemmKernelParams::valid_shape(1, 65536, 65536));
    static_assert(!DirectAMD::GemmKernelParams::valid_shape(65536, 65536, 1));

    DirectAMD::PM4Builder gemm;
    gemm.add_polaris_gemm(0x100003000ULL, {
        0x100000000ULL, 0x100001100ULL, 0x100002100ULL, 65, 129, 33
    });
    const auto& gemm_dispatch = gemm.get_packets();
    assert(gemm_dispatch.size() == 40);
    assert(gemm_dispatch[16] == 0xC0042);
    assert(gemm_dispatch[17] == 0x192); // Nine user SGPRs and group IDs X/Y
    assert(gemm_dispatch[24] == 0xC0097600);
    assert(gemm_dispatch[32] == 65);
    assert(gemm_dispatch[33] == 129);
    assert(gemm_dispatch[34] == 33);
    assert(gemm_dispatch[35] == 0xC0031502);
    assert(gemm_dispatch[36] == 3);
    assert(gemm_dispatch[37] == 65);
    assert(gemm_dispatch[38] == 1);

    static_assert(DirectAMD::SoftmaxKernelParams::valid_count(1));
    static_assert(DirectAMD::SoftmaxKernelParams::valid_count(1u << 30));
    static_assert(!DirectAMD::SoftmaxKernelParams::valid_count(0));
    static_assert(!DirectAMD::SoftmaxKernelParams::valid_count((1u << 30) + 1));

    DirectAMD::PM4Builder softmax;
    softmax.add_polaris_softmax(0x100003000ULL, {0x100000100ULL, 0x100001100ULL, 65537});
    const auto& softmax_dispatch = softmax.get_packets();
    assert(softmax_dispatch.size() == 40);
    assert(softmax_dispatch[16] == 0xF0042); // Twelve VGPRs and preserved FP32 subnormals
    assert(softmax_dispatch[17] == 0x8A);
    assert(softmax_dispatch[24] == 0xC0057600);
    assert(softmax_dispatch[26] == 0x100);
    assert(softmax_dispatch[28] == 0x1100);
    assert(softmax_dispatch[30] == 65537);
    assert(softmax_dispatch[31] == 0xC0031502);
    assert(softmax_dispatch[32] == 1);
    assert(softmax_dispatch[33] == 1);
    assert(softmax_dispatch[34] == 1);
    assert(softmax_dispatch[39] == 0xFFFF1000);

    std::cout << "[+] Serialized " << packets.size() << " dwords of PM4 compute packets.\n";
    std::cout << "[PASS] PM4 Packet Builder Test Completed.\n";
    return 0;
}

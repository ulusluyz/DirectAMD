#include "directamd/common.hpp"
#include "directamd/device.hpp"
#include "directamd/memory.hpp"
#include "directamd/command.hpp"
#include "directamd/isa.hpp"
#include "directamd/error.hpp"

#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "====================================================\n";
    std::cout << "          DirectAMD Vector-Add Prototype            \n";
    std::cout << "====================================================\n";

    // 1. CPU Reference Data
    const uint32_t N = 4;
    std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_B = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> h_C_expected = {6.0f, 8.0f, 10.0f, 12.0f};
    std::vector<float> h_C_gpu(N, 0.0f);

    std::cout << "[+] Host Input Vector A: [1, 2, 3, 4]\n";
    std::cout << "[+] Host Input Vector B: [5, 6, 7, 8]\n";

    // 2. Hardware Device Discovery
    std::error_code ec;
    auto device = DirectAMD::Device::open_first_available(ec);

    if (!device) {
        std::cout << "\n----------------------------------------------------\n";
        std::cout << "PHYSICAL GPU DETECTED: NO (/dev/dri/renderD* missing)\n";
        std::cout << "STATUS: MILESTONE 1: NOT TESTED — PHYSICAL AMD GPU REQUIRED\n";
        std::cout << "----------------------------------------------------\n";
        std::cout << "[+] Verifying CPU Reference Math:\n";
        for (size_t i = 0; i < N; ++i) {
            float sum = h_A[i] + h_B[i];
            std::cout << "    C[" << i << "] = " << sum << " (Expected: " << h_C_expected[i] << ")\n";
        }
        std::cout << "CPU REFERENCE PASS\n";
        return 0;
    }

    std::cout << "[+] AMD GPU Found at: " << device->get_properties().device_path << "\n";
    std::cout << "[+] Architecture: " << device->get_properties().arch_name << "\n";

    // 3. Context Creation
    uint32_t ctx_id = device->create_context(ec);
    if (ec) {
        std::cerr << "[-] Failed to create GPU context: " << ec.message() << "\n";
        return 1;
    }

    // 4. Memory Allocations (GTT/VRAM)
    DirectAMD::VAManager va_mgr;
    size_t size_bytes = N * sizeof(float);

    auto bo_A = DirectAMD::BufferObject::create(*device, va_mgr, size_bytes, 0x2 /* GTT */, ec);
    auto bo_B = DirectAMD::BufferObject::create(*device, va_mgr, size_bytes, 0x2 /* GTT */, ec);
    auto bo_C = DirectAMD::BufferObject::create(*device, va_mgr, size_bytes, 0x2 /* GTT */, ec);

    if (!bo_A || !bo_B || !bo_C) {
        std::cerr << "[-] BO Allocation failed: " << ec.message() << "\n";
        return 1;
    }

    bo_A->copy_to_gpu(h_A.data(), size_bytes, ec);
    bo_B->copy_to_gpu(h_B.data(), size_bytes, ec);

    // 5. ISA Machine Code Assembly
    auto kernel_code = DirectAMD::ISA::RDNAEncoder::emit_vector_add_kernel();
    size_t code_bytes = kernel_code.size() * sizeof(uint32_t);

    auto bo_code = DirectAMD::BufferObject::create(*device, va_mgr, code_bytes, 0x2, ec);
    bo_code->copy_to_gpu(kernel_code.data(), code_bytes, ec);

    // 6. PM4 Packet Builder
    DirectAMD::PM4Builder pm4;
    pm4.add_acquire_mem();

    // Set Kernel Program Address (COMPUTE_PGM_LO/HI: 0x2E0C / 0x2E0D)
    uint64_t code_va = bo_code->get_gpu_va();
    pm4.add_set_sh_reg(0x2E0C, {
        static_cast<uint32_t>(code_va >> 8),
        static_cast<uint32_t>(code_va >> 40)
    });

    // Set User SGPRs (COMPUTE_USER_DATA_0: 0x2E40)
    // s[0:1]=A_va, s[2:3]=B_va, s[4:5]=C_va, s6=N
    uint64_t va_a = bo_A->get_gpu_va();
    uint64_t va_b = bo_B->get_gpu_va();
    uint64_t va_c = bo_C->get_gpu_va();

    pm4.add_set_sh_reg(0x2E40, {
        static_cast<uint32_t>(va_a & 0xFFFFFFFF),
        static_cast<uint32_t>(va_a >> 32),
        static_cast<uint32_t>(va_b & 0xFFFFFFFF),
        static_cast<uint32_t>(va_b >> 32),
        static_cast<uint32_t>(va_c & 0xFFFFFFFF),
        static_cast<uint32_t>(va_c >> 32),
        N
    });

    // Set Threads per Workgroup: COMPUTE_NUM_THREAD_X/Y/Z (0x2E07)
    pm4.add_set_sh_reg(0x2E07, {64, 1, 1});

    // Dispatch Grid (1 workgroup)
    pm4.add_dispatch_direct(1, 1, 1);
    pm4.add_acquire_mem();

    auto bo_ib = DirectAMD::BufferObject::create(*device, va_mgr, pm4.get_size_bytes(), 0x2, ec);
    bo_ib->copy_to_gpu(pm4.get_packets().data(), pm4.get_size_bytes(), ec);

    // 7. CS Command Submission
    uint64_t fence = DirectAMD::CommandSubmission::submit_compute(
        *device, ctx_id, *bo_ib, {bo_A.get(), bo_B.get(), bo_C.get(), bo_code.get()}, ec);

    if (ec) {
        std::cerr << "[-] Command submission failed: " << ec.message() << "\n";
        return 1;
    }

    // 8. Synchronization
    bool complete = DirectAMD::CommandSubmission::wait_fence(*device, ctx_id, fence, 1000000000ULL, ec);
    if (!complete || ec) {
        std::cerr << "[-] Fence wait failed or timed out: " << ec.message() << "\n";
        return 1;
    }

    // 9. Read back result
    bo_C->copy_from_gpu(h_C_gpu.data(), size_bytes, ec);

    bool match = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(h_C_gpu[i] - h_C_expected[i]) > 1e-4f) {
            match = false;
        }
    }

    if (match) {
        std::cout << "MILESTONE 1: PASS\n";
    } else {
        std::cout << "MILESTONE 1: FAIL (Value mismatch)\n";
    }

    device->destroy_context(ctx_id, ec);
    return 0;
}

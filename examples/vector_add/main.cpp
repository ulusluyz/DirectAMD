#include "directamd/command.hpp"
#include "directamd/device.hpp"
#include "directamd/isa.hpp"
#include "directamd/memory.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <drm/amdgpu_drm.h>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

namespace {

bool run_vector_add(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va,
                    std::span<const float> a, std::span<const float> b) {
    constexpr uint32_t wave_size = DirectAMD::ISA::PolarisEncoder::wave_size;
    const uint32_t count = static_cast<uint32_t>(a.size());
    const size_t bytes = a.size_bytes();
    const size_t padded_count = (count + wave_size - 1) / wave_size * wave_size;
    constexpr float guard = -123456.0f;
    std::vector<float> output(padded_count, guard);
    std::error_code ec;

    const auto failure = [&](const char* step) {
        std::cerr << step << " failed: " << ec.message() << '\n';
        return false;
    };
    const auto make_buffer = [&](size_t size) {
        return DirectAMD::BufferObject::create(device, va, size, AMDGPU_GEM_DOMAIN_GTT, ec);
    };

    auto bo_a = make_buffer(bytes);
    if (!bo_a) return failure("Allocate A");
    auto bo_b = make_buffer(bytes);
    if (!bo_b) return failure("Allocate B");
    auto bo_c = make_buffer(output.size() * sizeof(float));
    if (!bo_c) return failure("Allocate C");
    bo_a->copy_to_gpu(a.data(), bytes, ec);
    if (ec) return failure("Upload A");
    bo_b->copy_to_gpu(b.data(), bytes, ec);
    if (ec) return failure("Upload B");
    bo_c->copy_to_gpu(output.data(), output.size() * sizeof(float), ec);
    if (ec) return failure("Initialize C");

    auto kernel = DirectAMD::ISA::PolarisEncoder::emit_vector_add_kernel();
    const size_t code_bytes = kernel.size() * sizeof(uint32_t);
    auto bo_code = make_buffer(code_bytes);
    if (!bo_code) return failure("Allocate kernel");
    bo_code->copy_to_gpu(kernel.data(), code_bytes, ec);
    if (ec) return failure("Upload kernel");

    DirectAMD::PM4Builder pm4;
    pm4.add_polaris_vector_add(bo_code->get_gpu_va(), {
        bo_a->get_gpu_va(), bo_b->get_gpu_va(), bo_c->get_gpu_va(), count
    });
    auto bo_ib = make_buffer(pm4.get_size_bytes());
    if (!bo_ib) return failure("Allocate command buffer");
    bo_ib->copy_to_gpu(pm4.get_packets().data(), pm4.get_size_bytes(), ec);
    if (ec) return failure("Upload command buffer");

    const uint64_t fence = DirectAMD::CommandSubmission::submit_compute(
        device, context, *bo_ib,
        {bo_a.get(), bo_b.get(), bo_c.get(), bo_code.get()}, ec);
    if (ec) return failure("Submit GPU command buffer");
    if (fence == 0) {
        std::cerr << "GPU submission returned no fence\n";
        return false;
    }

    constexpr auto timeout = std::chrono::nanoseconds(std::chrono::seconds(2));
    if (!DirectAMD::CommandSubmission::wait_fence(device, context, fence, timeout.count(), ec)) {
        return failure("Wait for GPU fence");
    }

    // Discard the host-side initialization so validation depends on successful readback.
    std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
    bo_c->copy_from_gpu(output.data(), output.size() * sizeof(float), ec);
    if (ec) return failure("Read GPU output");

    for (size_t i = 0; i < a.size(); ++i) {
        const float expected = a[i] + b[i];
        if (!std::isfinite(output[i]) || output[i] != expected) {
            std::cerr << "GPU C[" << i << "] = " << output[i] << ", expected " << expected << '\n';
            return false;
        }
    }
    for (size_t i = a.size(); i < output.size(); ++i) {
        if (output[i] != guard) {
            std::cerr << "GPU modified guard element C[" << i << "]\n";
            return false;
        }
    }

    std::cout << "GPU vector-add PASS: " << count << " results and "
              << output.size() - a.size() << " guard elements verified (fence " << fence << ")\n";
    if (count == 4) {
        std::cout << "GPU output: [" << output[0] << ", " << output[1] << ", "
                  << output[2] << ", " << output[3] << "]\n";
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [/dev/dri/renderDxxx]\n";
        return 2;
    }

    std::error_code ec;
    auto device = argc == 2 ? DirectAMD::Device::open_path(argv[1], ec)
                            : DirectAMD::Device::open_first_available(ec);
    if (!device) {
        std::cerr << "Cannot open AMD GPU";
        if (argc == 2) std::cerr << " " << argv[1];
        std::cerr << ": " << ec.message() << '\n';
        return 1;
    }

    const auto& properties = device->get_properties();
    std::cout << "AMD GPU: " << properties.device_path << " | " << properties.arch_name
              << " | " << properties.num_compute_units << " CUs"
              << " | wave" << properties.wave_front_size << '\n';
    if (properties.arch_family != DirectAMD::ArchitectureFamily::GCN4_POLARIS) {
        std::cerr << "This GPU vector-add example requires a Polaris 10 device\n";
        return 1;
    }

    const uint32_t context = device->create_context(ec);
    if (ec) {
        std::cerr << "Create GPU context failed: " << ec.message() << '\n';
        return 1;
    }

    DirectAMD::VAManager va;
    const float short_a[] = {1, 2, 3, 4};
    const float short_b[] = {5, 6, 7, 8};
    const bool short_passed = run_vector_add(*device, context, va, short_a, short_b);

    bool long_passed = false;
    if (short_passed) {
        std::vector<float> a(257);
        std::vector<float> b(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<float>(i) * 0.5f - 64.0f;
            b[i] = 72.25f - static_cast<float>(i) * 0.25f;
        }
        long_passed = run_vector_add(*device, context, va, a, b);
    }

    device->destroy_context(context, ec);
    if (ec) {
        std::cerr << "Destroy GPU context failed: " << ec.message() << '\n';
        return 1;
    }
    return short_passed && long_passed ? 0 : 1;
}

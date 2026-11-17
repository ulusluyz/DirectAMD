#include "directamd/device.hpp"
#include "directamd/memory.hpp"
#include "directamd/silu.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

double cpu_reference(double x) {
    if (x < 0.0) {
        const double t = std::exp(x);
        return x * t / (1.0 + t);
    }
    return x / (1.0 + std::exp(-x));
}

bool run_case(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va,
              std::string_view label, std::span<const float> input, bool in_place = false) {
    constexpr float sentinel = 345678.0f;
    std::vector<float> storage(input.size() + 2, sentinel);
    auto output = std::span(storage).subspan(1, input.size());
    auto gpu_input = input;
    if (in_place) {
        std::ranges::copy(input, output.begin());
        gpu_input = output;
    }

    std::error_code ec;
    if (!DirectAMD::silu_fp32(device, context, va, gpu_input, output, ec)) {
        std::cerr << label << " SiLU failed: " << ec.message() << '\n';
        return false;
    }
    if (storage.front() != sentinel || storage.back() != sentinel) {
        std::cerr << label << " SiLU wrote outside the host output span\n";
        return false;
    }

    double max_abs_error = 0.0;
    double max_relative_error = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        const double expected = cpu_reference(input[i]);
        const double error = std::abs(output[i] - expected);
        const double tolerance = 2e-5 * std::abs(expected) +
                                 0.51 * std::numeric_limits<float>::denorm_min();
        if (!std::isfinite(output[i]) || error > tolerance ||
            std::signbit(output[i]) != std::signbit(expected)) {
            std::cerr << label << " at index " << i << " for x=" << input[i]
                      << ": GPU " << output[i] << ", CPU " << expected
                      << ", tolerance " << tolerance << '\n';
            return false;
        }
        max_abs_error = std::max(max_abs_error, error);
        if (std::abs(expected) >= std::numeric_limits<float>::min()) {
            max_relative_error = std::max(max_relative_error, error / std::abs(expected));
        }
    }

    std::cout << "GPU SiLU PASS " << label << " | n=" << input.size()
              << " | in-place=" << (in_place ? "yes" : "no")
              << " | max abs error " << max_abs_error
              << " | max normal relative error " << max_relative_error
              << " | fence and output guards verified\n";
    return true;
}

bool invalid_inputs_rejected(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    const float valid[] = {1, -1};
    const float nan[] = {1, std::numeric_limits<float>::quiet_NaN()};
    const float pos_inf[] = {1, std::numeric_limits<float>::infinity()};
    const float neg_inf[] = {1, -std::numeric_limits<float>::infinity()};
    constexpr float sentinel = -5.0f;
    float output[] = {sentinel, sentinel};
    std::error_code ec;
    const auto rejected = [&](std::span<const float> x, std::span<float> y) {
        return !DirectAMD::silu_fp32(device, context, va, x, y, ec) &&
               ec == std::errc::invalid_argument;
    };
    if (!rejected({}, {}) || !rejected(valid, std::span(output).first(1)) ||
        !rejected(std::span(valid).first(1), output) ||
        !rejected(nan, output) || !rejected(pos_inf, output) || !rejected(neg_inf, output) ||
        output[0] != sentinel || output[1] != sentinel) {
        std::cerr << "SiLU did not reject invalid input cleanly\n";
        return false;
    }
    std::cout << "SiLU input validation PASS: empty, mismatched and nonfinite spans rejected\n";
    return true;
}

bool validate(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    if (!invalid_inputs_rejected(device, context, va)) return false;

    const float single[] = {-0.75f};
    const float zeros[] = {0.0f, -0.0f};
    constexpr float largest = std::numeric_limits<float>::max();
    constexpr float min_normal = std::numeric_limits<float>::min();
    constexpr float min_subnormal = std::numeric_limits<float>::denorm_min();
    const float extremes[] = {
        largest, -largest, 1e30f, -1e30f, min_normal, -min_normal,
        min_subnormal, -min_subnormal
    };
    const float tail[] = {-43, -45, -63, -83, -91, -95, -100, -106, -108, -110, -128, 128};
    if (!run_case(device, context, va, "single-negative", single) ||
        !run_case(device, context, va, "signed-zeros", zeros, true) ||
        !run_case(device, context, va, "FP32-extremes", extremes) ||
        !run_case(device, context, va, "negative-tail", tail)) {
        return false;
    }

    for (size_t length : {3, 7, 31, 63, 64, 65, 127, 257, 1024, 4097, 65537}) {
        std::vector<float> input(length);
        for (size_t i = 0; i < length; ++i) {
            input[i] = (static_cast<int>((i * 19 + i / 64 * 3) % 151) - 90) / 7.0f;
        }
        if (!run_case(device, context, va, "mixed", input, length == 65 || length == 65537)) {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " /dev/dri/renderDxxx\n";
        return 2;
    }
    std::error_code ec;
    auto device = DirectAMD::Device::open_path(argv[1], ec);
    if (!device) {
        std::cerr << "Cannot open " << argv[1] << ": " << ec.message() << '\n';
        return 1;
    }
    std::cout << "AMD GPU: " << device->get_properties().device_path << " | "
              << device->get_properties().arch_name << '\n';
    const uint32_t context = device->create_context(ec);
    if (ec) {
        std::cerr << "Cannot create GPU context: " << ec.message() << '\n';
        return 1;
    }

    DirectAMD::VAManager va;
    const bool passed = validate(*device, context, va);
    device->destroy_context(context, ec);
    if (ec) {
        std::cerr << "Cannot destroy GPU context: " << ec.message() << '\n';
        return 1;
    }
    return passed ? 0 : 1;
}

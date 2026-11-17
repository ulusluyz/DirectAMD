#include "directamd/device.hpp"
#include "directamd/memory.hpp"
#include "directamd/rmsnorm.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

bool run_case(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va,
              std::string_view label, std::span<const float> x, std::span<const float> weights,
              float epsilon) {
    constexpr float sentinel = 345678.0f;
    std::vector<float> storage(x.size() + 2, sentinel);
    auto output = std::span(storage).subspan(1, x.size());
    std::error_code ec;
    if (!DirectAMD::rmsnorm_fp32(device, context, va, x, weights, epsilon, output, ec)) {
        std::cerr << label << " RMSNorm failed: " << ec.message() << '\n';
        return false;
    }
    if (storage.front() != sentinel || storage.back() != sentinel) {
        std::cerr << label << " wrote outside the host output span\n";
        return false;
    }

    double squares = 0.0;
    for (double value : x) squares += value * value;
    const double denominator = std::sqrt(squares / static_cast<double>(x.size()) + epsilon);
    double max_error = 0.0;
    double max_relative_error = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double expected = static_cast<double>(x[i]) * weights[i] / denominator;
        const double error = std::abs(output[i] - expected);
        const double tolerance = 3e-6 * std::abs(expected) +
                                 2.0 * std::numeric_limits<float>::denorm_min();
        if (!std::isfinite(output[i]) || error > tolerance) {
            std::cerr << label << " at index " << i << ": GPU " << output[i]
                      << ", CPU " << expected << ", tolerance " << tolerance << '\n';
            return false;
        }
        max_error = std::max(max_error, error);
        if (std::abs(expected) >= std::numeric_limits<float>::min()) {
            max_relative_error = std::max(max_relative_error, error / std::abs(expected));
        }
    }
    std::cout << "GPU RMSNorm PASS " << label << " | n=" << x.size()
              << " | eps=" << epsilon << " | max abs error " << max_error
              << " | max normal relative error " << max_relative_error
              << " | fence and output guards verified\n";
    return true;
}

bool invalid_inputs_rejected(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    constexpr float sentinel = -5.0f;
    float output[] = {sentinel, sentinel};
    const float valid[] = {1.0f, -1.0f};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan_input[] = {1.0f, nan};
    const float infinite_weight[] = {1.0f, infinity};
    std::error_code ec;
    const auto rejected = [&](std::span<const float> x, std::span<const float> weights,
                              float epsilon, std::span<float> out) {
        return !DirectAMD::rmsnorm_fp32(device, context, va, x, weights, epsilon, out, ec) &&
               ec == std::errc::invalid_argument;
    };
    if (!rejected({}, {}, 1e-5f, {}) ||
        !rejected(valid, std::span(valid).first(1), 1e-5f, output) ||
        !rejected(valid, valid, 1e-5f, std::span(output).first(1)) ||
        !rejected(nan_input, valid, 1e-5f, output) ||
        !rejected(valid, infinite_weight, 1e-5f, output) ||
        !rejected(valid, valid, 0.0f, output) ||
        !rejected(valid, valid, -1e-5f, output) ||
        !rejected(valid, valid, nan, output) ||
        !rejected(valid, valid, infinity, output) ||
        output[0] != sentinel || output[1] != sentinel) {
        std::cerr << "RMSNorm did not reject invalid inputs cleanly\n";
        return false;
    }
    std::cout << "RMSNorm input validation PASS: empty/mismatched spans, nonfinite values and invalid epsilon rejected\n";
    return true;
}

bool validate(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    if (!invalid_inputs_rejected(device, context, va)) return false;

    constexpr float largest = std::numeric_limits<float>::max();
    constexpr float tiny = std::numeric_limits<float>::denorm_min();
    const float single[] = {-2.5f};
    const float single_weight[] = {0.75f};
    const float negative[] = {-1.0f, -2.25f, -0.03125f};
    const float signed_weights[] = {0.0f, -1.5f, 0.25f};
    const float zeros[] = {0.0f, -0.0f, 0.0f};
    const float extreme[] = {largest, -largest, 1e-33f};
    const float extreme_weights[] = {1.0f, 0.5f, largest};
    const float subnormal[] = {1.0f, 1e-39f, -1e-39f};
    const float ones[] = {1.0f, 1.0f, 1.0f};
    const float small[] = {1e-23f, -3e-24f, tiny};
    if (!run_case(device, context, va, "single", single, single_weight, 1e-5f) ||
        !run_case(device, context, va, "negative-fractional", negative, signed_weights, 0.125f) ||
        !run_case(device, context, va, "zero-subnormal-epsilon", zeros, ones, tiny) ||
        !run_case(device, context, va, "finite-extremes", extreme, extreme_weights, largest) ||
        !run_case(device, context, va, "subnormal-output", subnormal, ones, 1e-6f) ||
        !run_case(device, context, va, "tiny-input-and-epsilon", small, ones, tiny)) {
        return false;
    }

    struct Case { size_t length; float epsilon; };
    constexpr Case cases[] = {
        {7, 1e-6f}, {31, 0.25f}, {63, 1e-5f}, {64, 1e-6f}, {65, 0.1f},
        {127, 1e-9f}, {257, 1e-2f}, {1024, 2.0f}, {4097, 1e-5f}, {65537, 1e-6f}
    };
    for (const auto& [length, epsilon] : cases) {
        std::vector<float> x(length);
        std::vector<float> weights(length);
        for (size_t i = 0; i < length; ++i) {
            x[i] = (static_cast<int>((i * 19 + i / 64 * 3) % 101) - 60) / 7.0f;
            weights[i] = (static_cast<int>((i * 13 + 3) % 31) - 13) / 11.0f;
        }
        if (!run_case(device, context, va, "mixed", x, weights, epsilon)) return false;
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

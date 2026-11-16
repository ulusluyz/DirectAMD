#include "directamd/device.hpp"
#include "directamd/gemm.hpp"
#include "directamd/memory.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

struct Case {
    uint32_t rows;
    uint32_t columns;
    uint32_t inner;
    bool fractional = false;
};

bool run_case(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va,
              const Case& shape) {
    const auto [rows, columns, inner, fractional] = shape;
    std::vector<float> a(static_cast<size_t>(rows) * inner);
    std::vector<float> b(static_cast<size_t>(inner) * columns);
    for (size_t i = 0; i < a.size(); ++i) {
        const int value = static_cast<int>((5 + i * 17 + i / inner * 3) % 29) - 14;
        a[i] = static_cast<float>(value) / (fractional ? 13.0f : 8.0f);
    }
    for (size_t i = 0; i < b.size(); ++i) {
        const int value = static_cast<int>((3 + i * 11 + i / columns * 7) % 31) - 15;
        b[i] = static_cast<float>(value) / (fractional ? 17.0f : 4.0f);
    }

    constexpr float sentinel = 345678.0f;
    const size_t output_size = static_cast<size_t>(rows) * columns;
    std::vector<float> output(output_size + 2, sentinel);
    auto c = std::span(output).subspan(1, output_size);
    std::error_code ec;
    if (!DirectAMD::gemm_fp32(device, context, va, rows, columns, inner, a, b, c, ec)) {
        std::cerr << rows << 'x' << columns << 'x' << inner << " GEMM failed: " << ec.message() << '\n';
        return false;
    }
    if (output.front() != sentinel || output.back() != sentinel) {
        std::cerr << "GEMM wrote outside the host output span\n";
        return false;
    }

    double max_error = 0.0;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < columns; ++col) {
            double expected = 0.0;
            for (size_t k = 0; k < inner; ++k) {
                expected += static_cast<double>(a[row * inner + k]) * b[k * columns + col];
            }
            const float result = c[row * columns + col];
            const double error = std::abs(result - expected);
            const double tolerance = fractional ? 1e-5 + 2e-5 * std::abs(expected) : 0.0;
            if (!std::isfinite(result) || error > tolerance) {
                std::cerr << rows << 'x' << columns << 'x' << inner << " GEMM C[" << row
                          << ',' << col << "]: GPU " << result << ", CPU " << expected
                          << ", tolerance " << tolerance << '\n';
                return false;
            }
            max_error = std::max(max_error, error);
        }
    }
    std::cout << "GPU GEMM PASS " << rows << 'x' << columns << 'x' << inner
              << " (M/N/K) | " << c.size() << " results | max abs error " << max_error
              << " | fence and output guards verified\n";
    return true;
}

bool invalid_spans_rejected(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    const float a[] = {1, 2, 3, 4};
    const float b[] = {5, 6, 7, 8};
    constexpr float sentinel = -5.0f;
    float c[] = {sentinel, sentinel, sentinel, sentinel};
    std::error_code ec;
    const auto rejected = [&](std::span<const float> lhs, std::span<const float> rhs,
                              std::span<float> output) {
        return !DirectAMD::gemm_fp32(device, context, va, 2, 2, 2, lhs, rhs, output, ec) &&
               ec == std::errc::invalid_argument;
    };
    if (!rejected(std::span(a).first(3), b, c) ||
        !rejected(a, std::span(b).first(3), c) ||
        !rejected(a, b, std::span(c).first(3)) ||
        !std::ranges::all_of(c, [=](float value) { return value == sentinel; })) {
        std::cerr << "GEMM did not reject an undersized span cleanly\n";
        return false;
    }
    std::cout << "GEMM shape validation PASS: undersized A, B and C rejected\n";
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
    bool passed = invalid_spans_rejected(*device, context, va);
    constexpr Case cases[] = {
        {1, 1, 1}, {1, 129, 5}, {3, 5, 7}, {7, 1, 257}, {31, 63, 65},
        {64, 64, 64}, {65, 67, 17}, {17, 129, 33, true}, {129, 65, 97}, {67, 130, 259}
    };
    for (const auto& shape : cases) {
        if (!passed) break;
        passed = run_case(*device, context, va, shape);
    }

    device->destroy_context(context, ec);
    if (ec) {
        std::cerr << "Cannot destroy GPU context: " << ec.message() << '\n';
        return 1;
    }
    return passed ? 0 : 1;
}

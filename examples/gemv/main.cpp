#include "directamd/device.hpp"
#include "directamd/gemv.hpp"
#include "directamd/memory.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool run_case(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va,
              size_t rows, size_t columns, bool fractional) {
    std::vector<float> matrix(rows * columns);
    std::vector<float> x(columns);
    for (size_t i = 0; i < matrix.size(); ++i) {
        const int value = static_cast<int>((i * 17 + (i / columns) * 3) % 29) - 14;
        matrix[i] = static_cast<float>(value) / (fractional ? 13.0f : 8.0f);
    }
    for (size_t j = 0; j < columns; ++j) {
        const int value = static_cast<int>((j * 11 + 3) % 23) - 11;
        x[j] = static_cast<float>(value) / (fractional ? 17.0f : 4.0f);
    }

    constexpr float host_guard = 345678.0f;
    std::vector<float> output(rows + 2, host_guard);
    auto y = std::span(output).subspan(1, rows);
    std::error_code ec;
    if (!DirectAMD::gemv_fp32(device, context, va, matrix, x, y, ec)) {
        std::cerr << rows << 'x' << columns << " GEMV failed: " << ec.message() << '\n';
        return false;
    }
    if (output.front() != host_guard || output.back() != host_guard) {
        std::cerr << rows << 'x' << columns << " GEMV wrote outside the host output span\n";
        return false;
    }

    double max_error = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        double expected = 0.0;
        for (size_t j = 0; j < columns; ++j) {
            expected += static_cast<double>(matrix[i * columns + j]) * x[j];
        }
        const double error = std::abs(static_cast<double>(y[i]) - expected);
        const double tolerance = fractional ? 1e-5 + 2e-5 * std::abs(expected) : 0.0;
        if (!std::isfinite(y[i]) || error > tolerance) {
            std::cerr << rows << 'x' << columns << " GEMV row " << i << ": GPU " << y[i]
                      << ", CPU " << expected << ", tolerance " << tolerance << '\n';
            return false;
        }
        max_error = std::max(max_error, error);
    }
    std::cout << "GPU GEMV PASS " << rows << 'x' << columns << " | " << rows
              << " results | max abs error " << max_error << " | fence and output guards verified\n";
    return true;
}

bool invalid_shape_rejected(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    const float short_matrix[] = {1, 2, 3}; // 2x2 requires four elements.
    const float x[] = {1, 2};
    float y[] = {-5, -5};
    std::error_code ec;
    if (DirectAMD::gemv_fp32(device, context, va, short_matrix, x, y, ec) ||
        ec != std::errc::invalid_argument || y[0] != -5 || y[1] != -5) {
        std::cerr << "GEMV invalid shape was not rejected cleanly\n";
        return false;
    }
    std::cout << "GEMV shape validation PASS: undersized matrix rejected\n";
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
    bool passed = invalid_shape_rejected(*device, context, va);
    struct Case {
        size_t rows;
        size_t columns;
        bool fractional;
    };
    constexpr Case cases[] = {
        {1, 1, false}, {3, 5, false}, {63, 17, false}, {64, 64, false},
        {65, 129, true}, {127, 1, false}, {257, 73, false},
        {1, 4097, false}, {513, 1025, false}
    };
    for (const auto& [rows, columns, fractional] : cases) {
        if (!passed) break;
        passed = run_case(*device, context, va, rows, columns, fractional);
    }

    device->destroy_context(context, ec);
    if (ec) {
        std::cerr << "Cannot destroy GPU context: " << ec.message() << '\n';
        return 1;
    }
    return passed ? 0 : 1;
}

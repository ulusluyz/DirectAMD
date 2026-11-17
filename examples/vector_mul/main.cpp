#include "directamd/device.hpp"
#include "directamd/memory.hpp"
#include "directamd/vector_mul.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

enum class Alias { None, A, B };

uint32_t bits(float value) {
    return std::bit_cast<uint32_t>(value);
}

float rounded_fp32(double value) {
    // The product of two FP32 inputs is exact in binary64. Handle overflow
    // explicitly so the reference never narrows an out-of-range finite value.
    constexpr double largest = std::numeric_limits<float>::max();
    constexpr double overflow_threshold = largest + 0x1p103;
    const double magnitude = std::abs(value);
    if (magnitude >= overflow_threshold) {
        return std::signbit(value) ? -std::numeric_limits<float>::infinity()
                                   : std::numeric_limits<float>::infinity();
    }
    if (magnitude > largest) {
        return std::signbit(value) ? -std::numeric_limits<float>::max()
                                   : std::numeric_limits<float>::max();
    }
    return static_cast<float>(value);
}

bool run_case(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va,
              std::string_view label, std::span<const float> a, std::span<const float> b,
              Alias alias = Alias::None) {
    constexpr float sentinel = 345678.0f;
    std::vector<float> storage(a.size() + 2, sentinel);
    auto output = std::span(storage).subspan(1, a.size());
    auto gpu_a = a;
    auto gpu_b = b;
    if (alias == Alias::A) {
        std::ranges::copy(a, output.begin());
        gpu_a = output;
    } else if (alias == Alias::B) {
        std::ranges::copy(b, output.begin());
        gpu_b = output;
    }

    std::error_code ec;
    if (!DirectAMD::vector_mul_fp32(device, context, va, gpu_a, gpu_b, output, ec)) {
        std::cerr << label << " vector multiplication failed: " << ec.message() << '\n';
        return false;
    }
    if (storage.front() != sentinel || storage.back() != sentinel) {
        std::cerr << label << " wrote outside the host output span\n";
        return false;
    }

    double max_abs_error = 0.0;
    double max_relative_error = 0.0;
    size_t overflows = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double exact = static_cast<double>(a[i]) * b[i];
        const float expected = rounded_fp32(exact);
        if (bits(output[i]) != bits(expected)) {
            std::cerr << label << " index " << i << ": GPU " << output[i]
                      << " (0x" << std::hex << bits(output[i]) << "), CPU "
                      << std::dec << expected << " (0x" << std::hex << bits(expected)
                      << std::dec << ")\n";
            return false;
        }
        if (std::isinf(expected)) {
            ++overflows;
            continue;
        }
        const double error = std::abs(static_cast<double>(output[i]) - exact);
        max_abs_error = std::max(max_abs_error, error);
        if (std::isnormal(expected)) {
            max_relative_error = std::max(max_relative_error, error / std::abs(exact));
        }
    }

    const char* alias_name = alias == Alias::A ? "C=A" : alias == Alias::B ? "C=B" : "none";
    std::cout << "GPU vector-mul PASS " << label << " | n=" << a.size() << " | alias=" << alias_name
              << " | rounded FP32 bitwise match | overflow results " << overflows
              << " | max abs vs double " << max_abs_error
              << " | max normal relative vs double " << max_relative_error
              << " | fence and output guards verified\n";
    return true;
}

bool invalid_inputs_rejected(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    const float valid[] = {1.0f, -2.0f};
    const float nan[] = {1.0f, std::numeric_limits<float>::quiet_NaN()};
    const float infinity[] = {1.0f, std::numeric_limits<float>::infinity()};
    constexpr float sentinel = 345678.0f;
    float output[] = {sentinel, sentinel};
    std::error_code ec;
    const auto rejected = [&](std::span<const float> a, std::span<const float> b, std::span<float> c) {
        return !DirectAMD::vector_mul_fp32(device, context, va, a, b, c, ec) &&
               ec == std::errc::invalid_argument;
    };
    if (!rejected({}, {}, {}) || !rejected(valid, std::span(valid).first(1), output) ||
        !rejected(std::span(valid).first(1), valid, output) ||
        !rejected(valid, valid, std::span(output).first(1)) ||
        !rejected(nan, valid, output) || !rejected(valid, infinity, output) ||
        output[0] != sentinel || output[1] != sentinel) {
        std::cerr << "Vector multiplication did not reject invalid input cleanly\n";
        return false;
    }
    std::cout << "Vector-mul input validation PASS: empty, mismatched and nonfinite spans rejected\n";
    return true;
}

bool validate(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    if (!invalid_inputs_rejected(device, context, va)) return false;

    const float single_a[] = {-1.25f};
    const float single_b[] = {0.75f};
    const float zero_a[] = {0.0f, 0.0f, -0.0f, -0.0f};
    const float zero_b[] = {2.0f, -2.0f, 3.0f, -3.0f};
    constexpr float largest = std::numeric_limits<float>::max();
    constexpr float smallest = std::numeric_limits<float>::min();
    constexpr float tiny = std::numeric_limits<float>::denorm_min();
    constexpr std::array edge_a{
        largest, largest, -largest, largest, -largest, largest, smallest, smallest,
        tiny, -tiny, tiny, tiny, tiny, -smallest, 1e-20f, -1e-20f
    };
    const std::array edge_b{
        1.0f, 2.0f, 2.0f, 0.75f, -0.75f, std::nextafter(1.0f, 2.0f), 0.5f,
        std::nextafter(1.0f, 0.0f), 0.5f, 0.5f, 1.5f, 2.5f, 0x1p120f, 0.5f, 1e-20f, 1e-20f
    };
    if (!run_case(device, context, va, "single", single_a, single_b) ||
        !run_case(device, context, va, "signed-zeros", zero_a, zero_b, Alias::A) ||
        !run_case(device, context, va, "FP32-limits", edge_a, edge_b, Alias::B)) {
        return false;
    }

    for (size_t count : {2, 3, 7, 31, 63, 64, 65, 127, 257, 1024, 4097, 65537}) {
        std::vector<float> a(count), b(count);
        for (size_t i = 0; i < count; ++i) {
            a[i] = (static_cast<int>((19 * i + 3 * (i / 64)) % 151) - 90) / 7.0f;
            b[i] = (static_cast<int>((5 + 13 * i) % 31) - 13) / 11.0f;
        }
        const Alias alias = count == 65 ? Alias::A : count == 65537 ? Alias::B : Alias::None;
        if (!run_case(device, context, va, "mixed", a, b, alias)) return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " /dev/dri/renderDxxx\n";
        return 2;
    }
    if (std::fegetround() != FE_TONEAREST) {
        std::cerr << "The CPU reference requires round-to-nearest mode\n";
        return 1;
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

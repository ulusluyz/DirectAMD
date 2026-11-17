#include "directamd/device.hpp"
#include "directamd/memory.hpp"
#include "directamd/rope.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {

struct Case {
    uint32_t length;
    uint32_t rotary_dim;
    uint32_t position;
    float base;
    bool in_place = false;
};

bool run_case(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va,
              const Case& shape, DirectAMD::RopeLayout layout) {
    const auto [length, rotary_dim, position, base, in_place] = shape;
    const char* layout_name = layout == DirectAMD::RopeLayout::Interleaved ? "interleaved" : "half-split";
    const DirectAMD::RopeConfig config{position, rotary_dim, base, layout};

    std::vector<float> input(length);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = (static_cast<int>((i * 19 + i / 64 * 3) % 101) - 60) / 7.0f;
    }
    if (rotary_dim < length) input.back() = -0.0f;

    constexpr float sentinel = 345678.0f;
    std::vector<float> storage(length + 2, sentinel);
    auto output = std::span(storage).subspan(1, length);
    std::span<const float> gpu_input = input;
    if (in_place) {
        std::ranges::copy(input, output.begin());
        gpu_input = output;
    }

    std::error_code ec;
    if (!DirectAMD::rope_fp32(device, context, va, gpu_input, config, output, ec)) {
        std::cerr << layout_name << " RoPE failed for n=" << length
                  << ", rotary=" << rotary_dim << ", position=" << position << ": " << ec.message() << '\n';
        return false;
    }
    if (storage.front() != sentinel || storage.back() != sentinel) {
        std::cerr << "RoPE wrote outside the host output span\n";
        return false;
    }

    const uint32_t pairs = rotary_dim / 2;
    double max_error = 0.0;
    for (uint32_t pair = 0; pair < pairs; ++pair) {
        const bool interleaved = layout == DirectAMD::RopeLayout::Interleaved;
        const uint32_t i = interleaved ? 2 * pair : pair;
        const uint32_t j = interleaved ? i + 1 : pair + pairs;
        const double angle = static_cast<double>(position) /
                             std::pow(static_cast<double>(base), 2.0 * pair / rotary_dim);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        const double expected_i = input[i] * cosine - input[j] * sine;
        const double expected_j = input[i] * sine + input[j] * cosine;
        const double error = std::max(std::abs(output[i] - expected_i),
                                      std::abs(output[j] - expected_j));
        const double tolerance = 3e-6 * std::hypot(static_cast<double>(input[i]), input[j]) +
                                 2.0 * std::numeric_limits<float>::denorm_min();
        if (!std::isfinite(output[i]) || !std::isfinite(output[j]) || error > tolerance) {
            std::cerr << layout_name << " RoPE pair " << pair << " for n=" << length
                      << ", rotary=" << rotary_dim << ", position=" << position
                      << ": GPU [" << output[i] << ", " << output[j] << "], CPU ["
                      << expected_i << ", " << expected_j << "], tolerance " << tolerance << '\n';
            return false;
        }
        max_error = std::max(max_error, error);
    }
    for (size_t i = rotary_dim; i < length; ++i) {
        if (std::bit_cast<uint32_t>(output[i]) != std::bit_cast<uint32_t>(input[i])) {
            std::cerr << layout_name << " RoPE changed the unrotated element " << i << '\n';
            return false;
        }
    }
    std::cout << "GPU RoPE PASS " << layout_name << " | n=" << length << " | rotary=" << rotary_dim
              << " | position=" << position << " | base=" << base << " | max abs error " << max_error
              << " | suffix, fence and output guards verified\n";
    return true;
}

bool invalid_inputs_rejected(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    const float valid[] = {1, -1, 2, -2};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float nonfinite[] = {1, nan, 2, -2};
    constexpr float sentinel = -5.0f;
    float output[] = {sentinel, sentinel, sentinel, sentinel};
    const DirectAMD::RopeConfig good{7, 4};
    std::error_code ec;
    const auto rejected = [&](std::span<const float> x, DirectAMD::RopeConfig config,
                              std::span<float> y) {
        return !DirectAMD::rope_fp32(device, context, va, x, config, y, ec) &&
               ec == std::errc::invalid_argument;
    };

    if (!rejected({}, good, {}) || !rejected(valid, good, std::span(output).first(3)) ||
        !rejected(valid, {7, 0}, output) || !rejected(valid, {7, 3}, output) ||
        !rejected(valid, {7, 6}, output) || !rejected(valid, {7, 4, 0.5f}, output) ||
        !rejected(valid, {7, 4, nan}, output) || !rejected(valid, {7, 4, inf}, output) ||
        !rejected(valid, {7, 4, 10000, static_cast<DirectAMD::RopeLayout>(2)}, output) ||
        !rejected(nonfinite, good, output) ||
        !std::ranges::all_of(output, [](float value) { return value == sentinel; })) {
        std::cerr << "RoPE did not reject invalid inputs cleanly\n";
        return false;
    }
    std::cout << "RoPE input validation PASS: invalid spans, dimensions, bases, layout and input rejected\n";
    return true;
}

bool validate(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    if (!invalid_inputs_rejected(device, context, va)) return false;

    constexpr uint32_t max_position = std::numeric_limits<uint32_t>::max();
    constexpr Case cases[] = {
        {2, 2, 0, 10000}, {3, 2, 1, 10000}, {7, 6, 3, 10000},
        {63, 62, 11, 1}, {64, 64, 31, 10000}, {65, 64, 4096, 10000, true},
        {129, 128, 1234, 42}, {257, 130, 65535, 10000},
        {4097, 66, max_position, 10}, {65537, 65536, max_position, 500000}
    };
    for (auto layout : {DirectAMD::RopeLayout::Interleaved, DirectAMD::RopeLayout::HalfSplit}) {
        for (const auto& shape : cases) {
            if (!run_case(device, context, va, shape, layout)) return false;
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

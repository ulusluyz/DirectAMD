#include "directamd/device.hpp"
#include "directamd/memory.hpp"
#include "directamd/softmax.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <string_view>
#include <vector>

namespace {

std::vector<float> mixed_logits(size_t size, float shift = 0.0f) {
    std::vector<float> logits(size);
    for (size_t i = 0; i < size; ++i) {
        logits[i] = shift + (static_cast<int>((i * 19 + i / 64 * 3) % 101) - 60) * 0.25f;
    }
    return logits;
}

bool run_case(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va,
              std::string_view label, std::span<const float> input) {
    constexpr float sentinel = 345678.0f;
    std::vector<float> storage(input.size() + 2, sentinel);
    auto output = std::span(storage).subspan(1, input.size());
    std::error_code ec;
    if (!DirectAMD::softmax_fp32(device, context, va, input, output, ec)) {
        std::cerr << label << " Softmax failed: " << ec.message() << '\n';
        return false;
    }
    if (storage.front() != sentinel || storage.back() != sentinel) {
        std::cerr << label << " wrote outside the host output span\n";
        return false;
    }

    const double maximum = std::ranges::max(input);
    std::vector<double> reference(input.size());
    std::ranges::transform(input, reference.begin(),
                           [maximum](double value) { return std::exp(value - maximum); });
    const double denominator = std::accumulate(reference.begin(), reference.end(), 0.0);
    const double sum = std::accumulate(output.begin(), output.end(), 0.0);
    const double sum_error = std::abs(sum - 1.0);
    if (sum_error > 2e-5) {
        std::cerr << label << " GPU probabilities sum to " << sum << '\n';
        return false;
    }

    double max_error = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        const double expected = reference[i] / denominator;
        const double absolute_error = std::abs(output[i] - expected);
        constexpr double subnormal_slack = 2.0 * std::numeric_limits<float>::denorm_min();
        const double tolerance = 2e-5 * expected + subnormal_slack;
        if (!std::isfinite(output[i]) || absolute_error > tolerance) {
            std::cerr << label << " at index " << i << ": GPU " << output[i]
                      << ", CPU " << expected << ", tolerance " << tolerance << '\n';
            return false;
        }
        max_error = std::max(max_error, absolute_error);
    }
    std::cout << "GPU Softmax PASS " << label << " | n=" << input.size()
              << " | max abs error " << max_error << " | sum error " << sum_error
              << " | fence and output guards verified\n";
    return true;
}

bool invalid_inputs_rejected(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    constexpr float sentinel = -5.0f;
    float output[] = {sentinel, sentinel};
    const float valid[] = {1.0f, -1.0f};
    const float nan[] = {1.0f, std::numeric_limits<float>::quiet_NaN()};
    const float infinity[] = {1.0f, std::numeric_limits<float>::infinity()};
    std::error_code ec;
    const auto rejected = [&](std::span<const float> input, std::span<float> out) {
        return !DirectAMD::softmax_fp32(device, context, va, input, out, ec) &&
               ec == std::errc::invalid_argument;
    };
    if (!rejected({}, {}) || !rejected(valid, std::span(output).first(1)) ||
        !rejected(nan, output) || !rejected(infinity, output) ||
        output[0] != sentinel || output[1] != sentinel) {
        std::cerr << "Softmax did not reject invalid inputs cleanly\n";
        return false;
    }
    std::cout << "Softmax input validation PASS: empty, mismatched and nonfinite spans rejected\n";
    return true;
}

bool validate(DirectAMD::Device& device, uint32_t context, DirectAMD::VAManager& va) {
    if (!invalid_inputs_rejected(device, context, va)) return false;

    constexpr float largest = std::numeric_limits<float>::max();
    const float single[] = {-10000};
    const float negative[] = {-10003, -10002, -10001};
    const float positive[] = {10000, 9999};
    const float extreme[] = {-largest, largest, 0, largest, -1, -largest, 1};
    const float subnormal[] = {0, -87, -90, -100, -103, -104, -10000};
    if (!run_case(device, context, va, "single", single) ||
        !run_case(device, context, va, "all-negative", negative) ||
        !run_case(device, context, va, "large-positive", positive) ||
        !run_case(device, context, va, "FP32-extremes", extreme) ||
        !run_case(device, context, va, "subnormal-tail", subnormal)) {
        return false;
    }

    for (size_t size : {31, 63, 64, 65, 127, 257, 1024}) {
        if (!run_case(device, context, va, "mixed", mixed_logits(size))) return false;
    }
    if (!run_case(device, context, va, "shifted-mixed", mixed_logits(65, 10000.0f))) return false;

    auto last_max = mixed_logits(4097, -10000.0f);
    last_max.back() = -9989.0f;
    if (!run_case(device, context, va, "maximum-in-tail", last_max)) return false;

    if (!run_case(device, context, va, "equal-large-positive",
                  std::vector<float>(65537, 10000.0f))) return false;

    // Each lane accumulates small terms after accumulating 16 terms equal to one.
    std::vector<float> small_terms(65537, -14.0f);
    std::fill_n(small_terms.begin(), 1024, 0.0f);
    return run_case(device, context, va, "small-terms-after-large", small_terms);
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

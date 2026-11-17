#ifndef DIRECTAMD_COMMON_HPP
#define DIRECTAMD_COMMON_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace DirectAMD {

enum class ArchitectureFamily {
    UNKNOWN = 0,
    GCN5_VEGA,
    RDNA1,
    RDNA2,
    RDNA3,
    CDNA1,
    CDNA2,
    CDNA3,
    GCN4_POLARIS
};

struct DeviceProperties {
    std::string device_path;
    uint32_t pci_vendor_id{0};
    uint32_t pci_device_id{0};
    uint32_t pci_revision_id{0};
    uint32_t asic_family{0};
    ArchitectureFamily arch_family{ArchitectureFamily::UNKNOWN};
    std::string arch_name;

    uint64_t vram_size_bytes{0};
    uint64_t gtt_size_bytes{0};
    uint32_t num_compute_units{0};
    uint32_t max_engine_clock_mhz{0};
    uint32_t max_memory_clock_mhz{0};
    uint32_t wave_front_size{32};
};

struct VectorAddKernelParams {
    uint64_t buffer_a_va{0};
    uint64_t buffer_b_va{0};
    uint64_t buffer_c_va{0};
    uint32_t count{0};
};

struct GemvKernelParams {
    uint64_t matrix_va{0};
    uint64_t vector_va{0};
    uint64_t output_va{0};
    uint32_t rows{0};
    uint32_t columns{0};

    // The kernel uses 32-bit byte offsets within the contiguous row-major matrix.
    static constexpr uint64_t max_elements = uint64_t{1} << 30;
    static constexpr bool valid_shape(uint64_t rows, uint64_t columns) noexcept {
        return rows != 0 && columns != 0 && rows <= max_elements / columns;
    }
};

struct GemmKernelParams {
    uint64_t a_va{0};
    uint64_t b_va{0};
    uint64_t c_va{0};
    uint32_t rows{0};
    uint32_t columns{0};
    uint32_t inner{0};

    static constexpr bool valid_shape(uint64_t rows, uint64_t columns, uint64_t inner) noexcept {
        // Every matrix must fit the kernel's unsigned 32-bit byte offsets.
        return GemvKernelParams::valid_shape(rows, inner) &&
               GemvKernelParams::valid_shape(inner, columns) &&
               GemvKernelParams::valid_shape(rows, columns);
    }
};

struct SoftmaxKernelParams {
    uint64_t input_va{0};
    uint64_t output_va{0};
    uint32_t count{0};

    // The kernel uses unsigned 32-bit byte offsets.
    static constexpr bool valid_count(uint64_t count) noexcept {
        return count != 0 && count <= (uint64_t{1} << 30);
    }
};

struct RmsNormKernelParams {
    uint64_t input_va{0};
    uint64_t weights_va{0};
    uint64_t output_va{0};
    uint32_t count{0};
    float epsilon{0};

    // The kernel uses unsigned 32-bit byte offsets.
    static constexpr bool valid_count(uint64_t count) noexcept {
        return count != 0 && count <= (uint64_t{1} << 30);
    }
};

enum class RopeLayout : uint32_t {
    Interleaved = 0,
    HalfSplit = 1
};

struct RopeKernelParams {
    uint64_t input_va;
    uint64_t phase_va;
    uint64_t output_va;
    uint32_t count;
    uint32_t rotary_dim;
    RopeLayout layout;

    // Element offsets must fit in an unsigned 32-bit byte offset.
    static constexpr bool valid_shape(uint64_t count, uint32_t rotary_dim) noexcept {
        return rotary_dim != 0 && rotary_dim % 2 == 0 &&
               rotary_dim <= count && count <= (uint64_t{1} << 30);
    }

    static constexpr bool valid_layout(RopeLayout layout) noexcept {
        return layout == RopeLayout::Interleaved || layout == RopeLayout::HalfSplit;
    }
};

struct SiluKernelParams {
    uint64_t input_va;
    uint64_t output_va;
    uint32_t count;

    // The kernel uses an unsigned 32-bit byte offset per element.
    static constexpr bool valid_count(uint64_t count) noexcept {
        return count != 0 && count <= (uint64_t{1} << 30);
    }
};

struct VectorMulKernelParams {
    uint64_t a_va;
    uint64_t b_va;
    uint64_t c_va;
    uint32_t count;

    // Element addresses use unsigned 32-bit byte offsets.
    static constexpr bool valid_count(uint64_t count) noexcept {
        return count != 0 && count <= (uint64_t{1} << 30);
    }
};

} // namespace DirectAMD

#endif // DIRECTAMD_COMMON_HPP

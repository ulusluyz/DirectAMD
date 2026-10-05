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
    CDNA3
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

} // namespace DirectAMD

#endif // DIRECTAMD_COMMON_HPP

#include "directamd/device.hpp"
#include "directamd/error.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <drm/drm.h>
#include <drm/amdgpu_drm.h>

#include <filesystem>
#include <iostream>
#include <cstring>

namespace DirectAMD {

ArchitectureFamily map_asic_family_to_arch(uint32_t asic_family) {
    // AMDGPU ASIC Family IDs from kernel headers
    // FAMILY_NV = 143 (Navi 1x/2x/3x)
    // FAMILY_VI = 130
    // FAMILY_AI = 141 (Vega)
    switch (asic_family) {
        case 141: // FAMILY_AI
            return ArchitectureFamily::GCN5_VEGA;
        case 143: // FAMILY_NV / Navi / RDNA
            return ArchitectureFamily::RDNA2;
        default:
            return ArchitectureFamily::UNKNOWN;
    }
}

std::vector<std::string> Device::scan_render_nodes() {
    std::vector<std::string> nodes;
    std::string base_path = "/dev/dri";

    if (!std::filesystem::exists(base_path)) {
        return nodes;
    }

    for (const auto& entry : std::filesystem::directory_iterator(base_path)) {
        std::string filename = entry.path().filename().string();
        if (filename.rfind("renderD", 0) == 0) {
            nodes.push_back(entry.path().string());
        }
    }
    return nodes;
}

std::unique_ptr<Device> Device::open_first_available(std::error_code& ec) {
    auto nodes = scan_render_nodes();
    if (nodes.empty()) {
        ec = make_error_code(ErrorCode::DEVICE_NOT_FOUND);
        return nullptr;
    }

    for (const auto& node : nodes) {
        auto dev = open_path(node, ec);
        if (dev) {
            ec.clear();
            return dev;
        }
    }

    ec = make_error_code(ErrorCode::DEVICE_NOT_FOUND);
    return nullptr;
}

std::unique_ptr<Device> Device::open_path(const std::string& path, std::error_code& ec) {
    int fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        ec = make_error_code(ErrorCode::DRM_ERROR);
        return nullptr;
    }

    DeviceProperties props;
    props.device_path = path;

    // Direct C struct instantiation to query device properties via ioctl
    struct drm_amdgpu_info dev_info_cmd{};
    struct drm_amdgpu_info_device dev_info{};

    dev_info_cmd.return_pointer = reinterpret_cast<uint64_t>(&dev_info);
    dev_info_cmd.return_size = sizeof(dev_info);
    dev_info_cmd.query = AMDGPU_INFO_DEV_INFO;

    if (ioctl(fd, DRM_IOCTL_AMDGPU_INFO, &dev_info_cmd) == 0) {
        props.pci_device_id = dev_info.device_id;
        props.pci_revision_id = dev_info.pci_rev;
        props.asic_family = dev_info.family;
        props.num_compute_units = dev_info.cu_active_number;
        props.arch_family = map_asic_family_to_arch(dev_info.family);
        props.arch_name = (props.arch_family == ArchitectureFamily::RDNA2) ? "RDNA2/RDNA3" : "GCN/Unknown";
    }

    // Query VRAM & GTT sizes via AMDGPU_INFO_VRAM_GTT
    struct drm_amdgpu_info_vram_gtt vram_gtt{};
    dev_info_cmd.return_pointer = reinterpret_cast<uint64_t>(&vram_gtt);
    dev_info_cmd.return_size = sizeof(vram_gtt);
    dev_info_cmd.query = AMDGPU_INFO_VRAM_GTT;

    if (ioctl(fd, DRM_IOCTL_AMDGPU_INFO, &dev_info_cmd) == 0) {
        props.vram_size_bytes = vram_gtt.vram_size;
        props.gtt_size_bytes = vram_gtt.gtt_size;
    }

    ec.clear();
    return std::unique_ptr<Device>(new Device(fd, props));
}

Device::Device(int fd, DeviceProperties props) : fd_(fd), props_(std::move(props)) {}

Device::~Device() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

uint32_t Device::create_context(std::error_code& ec) {
    if (fd_ < 0) {
        ec = make_error_code(ErrorCode::DEVICE_NOT_FOUND);
        return 0;
    }

    union drm_amdgpu_ctx ctx_args{};
    ctx_args.in.op = AMDGPU_CTX_OP_ALLOC_CTX;

    int ret = ioctl(fd_, DRM_IOCTL_AMDGPU_CTX, &ctx_args);
    if (ret < 0) {
        ec = make_error_code(ErrorCode::DRM_ERROR);
        return 0;
    }

    ec.clear();
    return ctx_args.out.alloc.ctx_id;
}

void Device::destroy_context(uint32_t ctx_id, std::error_code& ec) {
    if (fd_ < 0) {
        ec = make_error_code(ErrorCode::DEVICE_NOT_FOUND);
        return;
    }

    union drm_amdgpu_ctx ctx_args{};
    ctx_args.in.op = AMDGPU_CTX_OP_FREE_CTX;
    ctx_args.in.ctx_id = ctx_id;

    if (ioctl(fd_, DRM_IOCTL_AMDGPU_CTX, &ctx_args) < 0) {
        ec = make_error_code(ErrorCode::DRM_ERROR);
        return;
    }
    ec.clear();
}

} // namespace DirectAMD

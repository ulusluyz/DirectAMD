#ifndef DIRECTAMD_DEVICE_HPP
#define DIRECTAMD_DEVICE_HPP

#include "directamd/common.hpp"
#include "directamd/error.hpp"
#include <string>
#include <vector>
#include <memory>

namespace DirectAMD {

class Device {
public:
    static std::vector<std::string> scan_render_nodes();
    static std::unique_ptr<Device> open_first_available(std::error_code& ec);
    static std::unique_ptr<Device> open_path(const std::string& path, std::error_code& ec);

    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    int get_fd() const noexcept { return fd_; }
    const DeviceProperties& get_properties() const noexcept { return props_; }
    bool is_valid() const noexcept { return fd_ >= 0; }

    uint32_t create_context(std::error_code& ec);
    void destroy_context(uint32_t ctx_id, std::error_code& ec);

private:
    explicit Device(int fd, DeviceProperties props);
    bool query_device_info();

    int fd_{-1};
    DeviceProperties props_;
};

ArchitectureFamily map_asic_family_to_arch(uint32_t asic_family);

} // namespace DirectAMD

#endif // DIRECTAMD_DEVICE_HPP

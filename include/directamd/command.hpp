#ifndef DIRECTAMD_COMMAND_HPP
#define DIRECTAMD_COMMAND_HPP

#include "directamd/common.hpp"
#include "directamd/error.hpp"
#include <cstdint>
#include <vector>

namespace DirectAMD {

class Device;
class BufferObject;

class PM4Builder {
public:
    static uint32_t build_type3_header(uint8_t opcode, uint16_t count);

    // Appends PM4 PACKET3_SET_SH_REG
    void add_set_sh_reg(uint32_t reg_offset, const std::vector<uint32_t>& values);

    // Appends PM4 PACKET3_ACQUIRE_MEM (Cache invalidate & flush)
    void add_acquire_mem(uint32_t engine = 0);

    // Appends PM4 PACKET3_DISPATCH_DIRECT
    void add_dispatch_direct(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t dispatch_initiator = 1);

    const std::vector<uint32_t>& get_packets() const noexcept { return packets_; }
    size_t get_size_bytes() const noexcept { return packets_.size() * sizeof(uint32_t); }

private:
    std::vector<uint32_t> packets_;
};

class CommandSubmission {
public:
    static uint64_t submit_compute(Device& device, uint32_t ctx_id,
                                    BufferObject& ib_bo,
                                    const std::vector<BufferObject*>& bo_handles,
                                    std::error_code& ec);

    static bool wait_fence(Device& device, uint32_t ctx_id, uint64_t fence_handle,
                           uint64_t timeout_ns, std::error_code& ec);
};

} // namespace DirectAMD

#endif // DIRECTAMD_COMMAND_HPP

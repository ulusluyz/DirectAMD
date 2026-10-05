#ifndef DIRECTAMD_ERROR_HPP
#define DIRECTAMD_ERROR_HPP

#include <string>
#include <system_error>

namespace DirectAMD {

enum class ErrorCode {
    SUCCESS = 0,
    DEVICE_NOT_FOUND,
    UNSUPPORTED_GPU,
    DRM_ERROR,
    MEMORY_ALLOCATION_FAILED,
    GPU_MAPPING_FAILED,
    COMMAND_SUBMISSION_FAILED,
    GPU_TIMEOUT,
    KERNEL_EXECUTION_FAILED,
    INVALID_MODEL,
    OUT_OF_MEMORY,
    HARDWARE_UNAVAILABLE
};

class DirectAMDErrorCategory : public std::error_category {
public:
    const char* name() const noexcept override;
    std::string message(int ev) const override;
    static const DirectAMDErrorCategory& get();
};

std::error_code make_error_code(ErrorCode e);

const char* error_code_to_string(ErrorCode code);

} // namespace DirectAMD

namespace std {
template <>
struct is_error_code_enum<DirectAMD::ErrorCode> : true_type {};
}

#endif // DIRECTAMD_ERROR_HPP

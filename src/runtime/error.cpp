#include "directamd/error.hpp"

namespace DirectAMD {

const char* DirectAMDErrorCategory::name() const noexcept {
    return "DirectAMD";
}

std::string DirectAMDErrorCategory::message(int ev) const {
    switch (static_cast<ErrorCode>(ev)) {
        case ErrorCode::SUCCESS:
            return "Success";
        case ErrorCode::DEVICE_NOT_FOUND:
            return "AMD GPU device node not found (/dev/dri/renderD*)";
        case ErrorCode::UNSUPPORTED_GPU:
            return "Unsupported AMD GPU architecture";
        case ErrorCode::DRM_ERROR:
            return "Linux DRM ioctl system error";
        case ErrorCode::MEMORY_ALLOCATION_FAILED:
            return "GPU GEM buffer object allocation failed";
        case ErrorCode::GPU_MAPPING_FAILED:
            return "GPU Virtual Address mapping failed";
        case ErrorCode::COMMAND_SUBMISSION_FAILED:
            return "AMDGPU Command Submission ioctl failed";
        case ErrorCode::GPU_TIMEOUT:
            return "GPU execution timed out";
        case ErrorCode::KERNEL_EXECUTION_FAILED:
            return "GPU kernel execution failed";
        case ErrorCode::INVALID_MODEL:
            return "Invalid model weights or configuration";
        case ErrorCode::OUT_OF_MEMORY:
            return "Out of GPU memory";
        case ErrorCode::HARDWARE_UNAVAILABLE:
            return "Physical AMD GPU hardware is unavailable (NOT TESTED — PHYSICAL AMD GPU REQUIRED)";
        default:
            return "Unknown DirectAMD error";
    }
}

const DirectAMDErrorCategory& DirectAMDErrorCategory::get() {
    static DirectAMDErrorCategory instance;
    return instance;
}

std::error_code make_error_code(ErrorCode e) {
    return {static_cast<int>(e), DirectAMDErrorCategory::get()};
}

const char* error_code_to_string(ErrorCode code) {
    switch (code) {
        case ErrorCode::SUCCESS: return "SUCCESS";
        case ErrorCode::DEVICE_NOT_FOUND: return "DEVICE_NOT_FOUND";
        case ErrorCode::UNSUPPORTED_GPU: return "UNSUPPORTED_GPU";
        case ErrorCode::DRM_ERROR: return "DRM_ERROR";
        case ErrorCode::MEMORY_ALLOCATION_FAILED: return "MEMORY_ALLOCATION_FAILED";
        case ErrorCode::GPU_MAPPING_FAILED: return "GPU_MAPPING_FAILED";
        case ErrorCode::COMMAND_SUBMISSION_FAILED: return "COMMAND_SUBMISSION_FAILED";
        case ErrorCode::GPU_TIMEOUT: return "GPU_TIMEOUT";
        case ErrorCode::KERNEL_EXECUTION_FAILED: return "KERNEL_EXECUTION_FAILED";
        case ErrorCode::INVALID_MODEL: return "INVALID_MODEL";
        case ErrorCode::OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case ErrorCode::HARDWARE_UNAVAILABLE: return "HARDWARE_UNAVAILABLE";
        default: return "UNKNOWN_ERROR";
    }
}

} // namespace DirectAMD

# DirectAMD Dependency Audit Report

## Overview
DirectAMD strictly prohibits external GPU compute runtimes, wrappers, or vendor stack dependencies. This document lists all permitted system dependencies and verifies the total absence of forbidden frameworks.

---

## Forbidden Technology Verification Matrix

| Technology | Status in DirectAMD | Verification Method |
| :--- | :--- | :--- |
| **CUDA** | **NOT USED** | Audited CMake, C++ includes, dynamic links |
| **ROCm** | **NOT USED** | Audited CMake, C++ includes, dynamic links |
| **HIP** | **NOT USED** | Audited CMake, C++ includes, dynamic links |
| **Tensor Core** | **NOT USED** | Audited code and ISA generators |
| **Vulkan / Vulkan Compute** | **NOT USED** | Audited CMake and headers |
| **OpenCL** | **NOT USED** | Audited CMake and headers |
| **cuBLAS / cuDNN** | **NOT USED** | Audited CMake and headers |
| **PyTorch / TensorFlow GPU Backend** | **NOT USED** | Standalone runtime, zero ML framework ties |
| **Other Runtime Wrappers** | **NOT USED** | Built strictly on Linux AMDGPU DRM UAPI |

---

## Allowed Low-Level Linux Dependencies

1. **Linux System Headers:** `<sys/ioctl.h>`, `<sys/mman.h>`, `<fcntl.h>`, `<unistd.h>`
2. **Linux DRM Kernel Headers:** `<drm/drm.h>`, `<drm/amdgpu_drm.h>`
3. **C++ Standard Library:** C++20 (`<vector>`, `<memory>`, `<cstdint>`, `<string>`, `<iostream>`, `<thread>`, `<atomic>`)
4. **Build System:** CMake (>= 3.20)

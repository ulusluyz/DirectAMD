# Contributing to DirectAMD

Thank you for your interest in DirectAMD! DirectAMD is a bare-metal C++20 experimental compute runtime built directly on the Linux AMDGPU DRM Kernel UAPI.

---

## Strict Scope & Forbidden Dependencies

DirectAMD is strictly an independent research project. Pull Requests adding dependencies or bindings for any of the following will be rejected:
* **CUDA**
* **ROCm**
* **HIP**
* **Vulkan / Vulkan Compute**
* **OpenCL**
* **cuBLAS / cuDNN**
* **PyTorch / TensorFlow backends**

All GPU interaction MUST occur via direct Linux DRM ioctls on `/dev/dri/renderD*`.

---

## Code Quality & Style Rules

1. **Language Standard:** Modern C++20.
2. **Compiler Warning Standards:** All builds must compile cleanly without warnings using `-Wall -Wextra -Wpedantic -Werror`.
3. **RAII & Safety:** Bare pointers for raw resources are forbidden. All GEM handles, mapped pointers, and file descriptors must be wrapped in deterministic RAII types.
4. **No Fake Hardware Reporting:** If testing on systems without physical AMD GPUs, tests MUST clearly report `SKIPPED — PHYSICAL AMD GPU REQUIRED`. Never substitute CPU fallback outputs as GPU results.

---

## Pull Request Submission Requirements

1. Verify build and CTest suite pass locally:
   ```bash
   mkdir -p build && cd build
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   cmake --build .
   ctest --output-on-failure
   ```
2. Ensure code follows formatting guidelines in `.editorconfig`.

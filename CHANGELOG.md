# Changelog

All notable changes to DirectAMD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

- Fixed the AMDGPU CS chunk address layout, indirect-buffer length and
  fence deadline.
- Added the Polaris 10 wave64 vector-add kernel and GFX8 PM4 dispatch.
  The example accepts a render-node path for per-card validation.
- Verified direct FP32 vector addition on the Radeon RX 570 and RX 580 and
  recorded the separate host and hardware results in the
  [validation report](docs/FINAL_VALIDATION_REPORT.md).

- Added row-major FP32 GEMV and GEMM APIs and per-device examples for
  Polaris 10. Both use guarded GPU readback through the existing DRM path.
- Revalidated vector add, GEMV and GEMM on the Radeon RX 570 and RX 580.

- Added stable FP32 Softmax on Polaris 10 and a per-device validation
  example. Verified it alongside the vector-add, GEMV and GEMM
  regressions on the Radeon RX 570 and RX 580.

- Added weighted FP32 RMSNorm and a per-device example for Polaris 10.
  Verified it and the vector-add, GEMV, GEMM and Softmax regressions on
  Radeon RX 570 and RX 580.

## [0.1.0] - 2026-10-05

### Added
- **Feasibility & Architecture Documentation:** Complete technical feasibility analysis and DRM UAPI specification in `docs/`.
- **Device Discovery Subsystem:** Linux DRM render node (`/dev/dri/renderD*`) scanner and `DRM_IOCTL_AMDGPU_INFO` device capability query module (`src/device/`).
- **Memory Subsystem:** Buffer Object Manager supporting VRAM (`AMDGPU_GEM_DOMAIN_VRAM`) and GTT (`AMDGPU_GEM_DOMAIN_GTT`) allocations, host CPU mmap bindings, and 48-bit GPU Virtual Address space management (`src/memory/`).
- **ISA Encoder Subsystem:** Zero-dependency C++ instruction synthesizer for RDNA2/RDNA3 & GCN compute instructions (`src/isa/`).
- **Command Subsystem:** Context creator, PM4 compute packet builder (`COMPUTE_PGM_LO/HI`, `PACKET3_DISPATCH_DIRECT`), and `DRM_IOCTL_AMDGPU_CS` submission manager (`src/command/`).
- **Build Infrastructure:** Warning-clean CMake build system, CTest suite, example programs, and GitHub Actions CI workflow.
- **Validation Reporting:** Strict Anti-Fake Validation Rule enforcement reporting `MILESTONE 1: NOT TESTED — PHYSICAL AMD GPU REQUIRED` when physical AMD GPU node is unavailable.

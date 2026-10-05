# DirectAMD Benchmarking Specification

## Policy Architecture
Benchmarking hardware performance metrics (e.g. GFLOPS, memory bandwidth in GB/s, latency in microseconds) requires execution on physical AMD GPU hardware.

* **No Synthetic / Mocked Benchmarks:** Benchmark numbers must never be generated from CPU emulators or arbitrary estimations.
* **Environment Reporting:** All benchmark results published in DirectAMD documentation MUST disclose exact hardware specs (GPU Model, Driver Version, Linux Kernel, Host CPU).

---

## Benchmark Metrics Plan (For Physical Hardware Validation)

When running on physical hardware, `benchmarks/` measures:

1. **Kernel Execution Time ($\mu s$):** Time elapsed between `DRM_IOCTL_AMDGPU_CS` submission and `DRM_IOCTL_AMDGPU_WAIT_CS` fence resolution.
2. **Buffer Transfer Bandwidth (GB/s):** PCIe Host-to-Device ($H \rightarrow D$) and Device-to-Host ($D \rightarrow H$) throughput over mapped GTT/VRAM buffers.
3. **Compute Throughput (GFLOPS):** Floating-point operations per second achieved during kernel execution.

# DirectAMD Roadmap

## Roadmap Stages

### Stage 1: Feasibility & Foundation
- [x] AMDGPU DRM / UAPI Technical Feasibility Research
- [x] Low-level Linux DRM UAPI Kernel Interface Specification
- [x] AMD GPU ISA & PM4 Command Buffer Research
- [x] Repository Skeleton & C++20 CMake Infrastructure Setup
- [x] Device Discovery Subsystem (`src/device/`)
- [x] GEM BO & Virtual Address Memory Manager (`src/memory/`)
- [x] PM4 Command Buffer & Context Submission Subsystem (`src/command/`)
- [x] Internal Zero-Dependency C++ ISA Encoder (`src/isa/`)
- [x] Vector-Add Prototype (`examples/vector_add/`)
- [x] Unit Test Suite & Hardware Validation Reporting (`tests/`)

### Stage 2: Physical Hardware Validation
- [x] Physical AMD GPU vector-add submission on RX 570 and RX 580
- [x] Compute fence synchronization and GPU readback verification
- [ ] Evaluation & Performance Logging

### Stage 3: Tensor Operations Subsystem
- [ ] Elementwise Vector Multiplication & Reduction
- [x] FP32 GEMV on Polaris 10, verified on RX 570 and RX 580
- [x] FP32 GEMM on Polaris 10, verified on RX 570 and RX 580
- [x] FP32 Softmax on Polaris 10, verified on RX 570 and RX 580
- [x] FP32 RMSNorm on Polaris 10, verified on RX 570 and RX 580
- [ ] RoPE, SiLU Compute Kernels

### Stage 4: Transformer LLM Inference Subsystem
- [ ] Memory Allocation for Transformer Weights & KV Cache
- [ ] Token Embeddings, Attention, Causal Masking, SwiGLU MLP
- [ ] Autoregressive Token Generation Pipeline

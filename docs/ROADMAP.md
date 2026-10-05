# DirectAMD Roadmap

## Roadmap Stages

### Stage 1: Feasibility & Foundation (Current Milestone)
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

### Stage 2: Physical Hardware Validation (Upon AMD Hardware Availability)
- [ ] Physical AMD GPU Hardware Vector-Add Kernel Submission
- [ ] Sequence Fence Synchronization & Verification
- [ ] Evaluation & Performance Logging

### Stage 3: Tensor Operations Subsystem
- [ ] Elementwise Vector Multiplication & Reduction
- [ ] FP32 GEMV / GEMM Matrix Multiplication Kernels
- [ ] RMSNorm, Softmax, RoPE, SiLU Compute Kernels

### Stage 4: Transformer LLM Inference Subsystem
- [ ] Memory Allocation for Transformer Weights & KV Cache
- [ ] Token Embeddings, Attention, Causal Masking, SwiGLU MLP
- [ ] Autoregressive Token Generation Pipeline

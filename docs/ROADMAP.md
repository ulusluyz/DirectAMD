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
- [x] FP32 elementwise vector multiplication on Polaris 10, verified on RX 570 and RX 580
- [ ] Reduction
- [x] FP32 GEMV on Polaris 10, verified on RX 570 and RX 580
- [x] FP32 GEMM on Polaris 10, verified on RX 570 and RX 580
- [x] FP32 Softmax on Polaris 10, verified on RX 570 and RX 580
- [x] FP32 RMSNorm on Polaris 10, verified on RX 570 and RX 580
- [x] FP32 RoPE (interleaved and half-split) on Polaris 10, verified on RX 570 and RX 580
- [x] FP32 SiLU on Polaris 10, verified on RX 570 and RX 580

### Stage 4: Transformer LLM Inference Subsystem
- [x] Persistent VRAM allocation for Transformer weights and causal GQA KV cache
- [x] GPU token embeddings, causal GQA attention and SwiGLU MLP using the existing primitives
- [x] Serial prefill and greedy autoregressive decode with EOS handling
- [x] Local Qwen2 SafeTensors/configuration and byte-level BPE/chat loading
- [x] Streaming Qwen2 GGUF F32/F16/Q4_K/Q6_K to FP16 SafeTensors conversion
- [x] Real converted Qwen2.5-1.5B Turkish response on RX 570 and RX 580
- [ ] Full 64-token real-model continuation validation (the tested question reached EOS at 11)

### Stage 5: Inference Performance (not started)
- [ ] Reduce primitive dispatch and fence overhead
- [ ] Improve GEMV throughput and serial prefill latency
- [ ] Revalidate numerical accuracy, memory bounds and existing GPU operations after optimization

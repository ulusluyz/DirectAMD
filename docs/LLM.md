# Local Qwen2 Transformer inference

The local CLI implements Qwen2 causal FP32 inference through DirectAMD's
AMDGPU/DRM kernels. It loads SafeTensors weights, constructs persistent
GPU tensors, runs serial prompt prefill and autoregressive greedy decode,
and stops at EOS or the requested token limit.

## Build and run

Python 3 and the PCRE2 8-bit runtime are required for the Qwen byte-level
BPE tokenizer. The Python modules use the standard library.

```bash
cmake -S . -B build-llm -DCMAKE_BUILD_TYPE=Debug -DENABLE_WERROR=ON
cmake --build build-llm --parallel
./build-llm/directamd-llm --model /local/qwen2.5-1.5b-fp16 \
  --device /dev/dri/renderD128 --prompt "Türkiye'nin başkenti neresidir?" \
  --context-length 160 --max-tokens 64 --memory-mib 7400
```

The model directory must contain config.json, tokenizer.json,
tokenizer_config.json and local SafeTensors weights. Sharded weights
use model.safetensors.index.json. F32, F16 and BF16 weights are converted
to FP32 during loading. The reader checks shapes, offsets, shard paths,
overlapping spans and the memory budget. It never downloads model files.

The loader targets standard Qwen2 architecture with SiLU, full causal
attention, half-split RoPE and Q/K/V projection biases. It supports tied
embeddings and GQA. The tokenizer supports Qwen2 byte-level BPE and the
standard Qwen2.5 system/user/assistant chat format for a single user
prompt. Tool-call execution, alternate tokenizers, direct quantized GGUF inference,
sliding-window attention and RoPE scaling require further implementation.

## GPU execution

Embedding rows are gathered by a GPU copy through the existing
elementwise-add path. Projections and logits use GEMV. Attention score
matrices use GEMV, scaling uses vector multiplication, and probabilities
use Softmax. Attention value aggregation uses GEMM. RMSNorm, RoPE, SiLU,
SwiGLU multiplication and residual additions reuse their existing kernels.

Weights and the per-layer, per-KV-head cache use persistent VRAM BOs.
Activations use GPU-visible GTT BOs. Cache updates are GPU copies.
Attention exposes only the cache prefix through the current position,
which enforces causality. Each dispatch waits for its compute fence.
All tensors have 64 sentinel elements on either side, checked around
dispatches and readback. Buffer views check bounds. The GPU session
rejects further operations after a failed submission or fence.

Decode currently uses synchronous primitive dispatches, and prompt
prefill processes tokens sequentially. These choices prioritize
arithmetic validation. The existing scalar-per-row GEMV kernel has
limited throughput for full language-model weight matrices.

## Local validation

```bash
ctest --test-dir build-llm --output-on-failure
./build-llm/directamd-llm --self-test --device /dev/dri/renderD128
./build-llm/directamd-llm --self-test --device /dev/dri/renderD129
```

The self-test creates temporary, deterministic, untrained SafeTensors
weights and compares GPU layer outputs with an independent double
precision Qwen2 implementation. It exercises F32/F16/BF16 conversion,
Unicode BPE/chat formatting, causal GQA, a 64-token greedy decode,
EOS, cache replay and bounds. A second synthetic model exercises
Qwen2.5-0.5B's head geometry (896 hidden dimensions, 14 query heads,
2 KV heads, head width 64).

Synthetic token IDs and timings describe the synthetic fixtures.
The supplied local Qwen2.5-1.5B GGUF can now be converted to FP16
SafeTensors with `directamd-convert`. See [GGUF conversion](GGUF_CONVERSION.md)
for supported formats, memory limits and the real-model command. Synthetic
self-test timings remain separate from trained-model performance.

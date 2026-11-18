# Local GGUF to SafeTensors conversion

`directamd-convert` reads an existing local GGUF v2/v3 Qwen2 model and writes
FP16 SafeTensors for the DirectAMD FP32 inference engine. Conversion uses CPU
code and bounded chunks of at most 262,144 elements. It does not download
models or call any GPU compute framework.

```bash
cmake -S . -B build-llm
cmake --build build-llm --parallel 2
./build-llm/directamd-convert \
  --input /home/cgu/Belgeler/qwen2.5-1.5b-instruct-q4_k_m.gguf --inspect
./build-llm/directamd-convert \
  --input /home/cgu/Belgeler/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --output /home/cgu/work/models/qwen2.5-1.5b-directamd-fp16-new
```

The output directory must not already exist. The converter checks available
disk space and leaves an `INCOMPLETE` marker until weights, configuration,
tokenizer and output tensor directory validation finish. The inference loader
rejects incomplete conversions. The source file is opened read-only.

Supported tensor types are F32 (0), F16 (1), Q4_K (12) and Q6_K (14).
Other types produce an explicit error before output creation. Q4_K_M is a
quantization preset, not a tensor type. The supplied file contains 169 Q4_K,
29 Q6_K and 141 F32 tensors. Q4_K uses 256-element/144-byte blocks; Q6_K uses
256-element/210-byte blocks, including signed per-group scales.

GGUF dimensions are stored fastest-first. Reversing the dimension list gives
SafeTensors row-major shapes without transposing the payload. Qwen2 uses
half-split RoPE; these Q/K matrices do not require an interleaved-RoPE
permutation. Tensor names map to the `model.layers.*`, embedding, final norm
and `lm_head` Qwen2 names. Missing, duplicate, out-of-bounds, misaligned,
wrong-shaped and unknown tensors are rejected. Nonfinite values and FP16
overflow are errors.

Configuration, BPE merges, token IDs, special tokens, chat template, GQA
head counts and RoPE base come from GGUF metadata. Only the Qwen2/gpt2
byte-level tokenizer preset is supported. The supplied vocabulary contains
271 UNUSED padding rows; their IDs are preserved. Tool-call template text is
preserved in the file, while the current inference CLI supports the standard
single-user Qwen chat path. Sliding-window attention, scaled RoPE and alternate
head dimensions are rejected.

Dequantization cannot recover information lost during Q4/Q6 quantization.
The FP16 output represents the decoded quantized model, rounded to FP16;
it does not reproduce the original unquantized training checkpoint.

## GPU memory

The supplied file retains separate embedding and output matrices with different
quantization. They are kept separate rather than incorrectly aliased. Its
1,777,088,000 stored parameters require 3,554,176,000 FP16 data bytes on disk
and 7,108,352,000 FP32 data bytes on the GPU, before guards, padding and cache.

LLM VRAM BOs are device-local and are not CPU-mapped. Existing GPU vector-add
with a zero operand transfers bounded GTT staging chunks. This avoids forcing
large weights through Polaris's 256 MiB CPU-visible VRAM aperture. Guard checks
also use staged GPU readback. The general BufferObject API keeps its previous
host-accessible default for existing operations.

`--memory-mib 0` is the CLI default and caps its tensor budget at device VRAM
capacity minus 256 MiB. An explicit smaller limit is still supported. The
loader reserves additional scratch space and validates weights plus KV cache
against the budget before allocation. Other applications also consume VRAM;
the budget does not reserve memory against those applications.

```bash
./build-llm/directamd-llm \
  --model /home/cgu/work/models/qwen2.5-1.5b-directamd-fp16 \
  --device /dev/dri/renderD128 \
  --prompt "Türkiye'nin başkenti neresidir?" \
  --context-length 160 --max-tokens 64 --memory-mib 7400
```

`max-tokens` is an upper bound. EOS can finish a response sooner. Loading time
is separate from first-token latency, which includes sequential prompt prefill.
The synchronous scalar-per-row GEMV implementation is slow on this model.

Format references are the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
and [GGML quantization layouts](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).

## Repeatable independent trained-model validation

```bash
g++ -O3 -ffp-contract=off -shared -fPIC llm/reference_fp16.cpp \
  -o build-llm/llm/libdirectamd_llm_reference.so
python3 llm/validate_trained.py \
  --model /home/cgu/work/models/qwen2.5-1.5b-directamd-fp16 \
  --reference-library build-llm/llm/libdirectamd_llm_reference.so \
  --backend build-llm/llm/libdirectamd_llm_backend.so \
  --device /dev/dri/renderD129
```

This optional check uses FP16 weights, mmap reads and independent CPU double
precision dot products. It compares two real prompt tokens across all 28
layers and logits. It does not validate every generated token. See the
[local hardware report](GGUF_VALIDATION.md) for the measured scope, timings,
errors, guard/fence counts and EOS behavior.

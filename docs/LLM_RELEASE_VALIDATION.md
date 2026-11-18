# Qwen2 inference and GGUF converter release validation

Validated locally on Debian 13, 18 November 2026, before publication.
This report describes executed tests, separately from the earlier development
measurements in [GGUF_VALIDATION.md](GGUF_VALIDATION.md).

## Host and converter checks

- The working project rebuilt successfully and CTest passed 7/7.
- The exact source overlay prepared on the current GitHub default-branch base
  also built from an empty build directory with Debug and ENABLE_WERROR=ON;
  its CTest suite passed 7/7.
- `test_gguf_converter` ran eight unittest cases, including full small GGUF
  conversion, output shape/type validation, incomplete-output rejection,
  quantization layouts, nonfinite/overflow rejection, input bounds and source
  preservation. `test_llm_host` ran its five loader/config/tokenizer tests.
- The actual local GGUF was inspected again: version 3, 339 tensors, 169 Q4_K,
  29 Q6_K and 141 F32, with Qwen2 H=1536, L=28, Q=12, KV=2 and D=128.
- Across 6048 sampled blocks / 912063 elements, both the converter's FP16
  output and written SafeTensors samples matched the independently installed
  GGML CPU dequantizer bit for bit; F32 used Python struct as the reference.
  This is sampled independent comparison, not exhaustive comparison of all blocks.
- The complete 339-tensor FP16 model was loaded into FP32 VRAM for both the
  production CLI run and the trained CPU/GPU reference check. Model files
  remain local and are not part of the Git tree.

## Eight GPU operations, separate from CTest

Both the working build and the clean release build reran all eight examples
on each physical GPU. The release-build result is tabulated below; duplicate
working-build runs are not added to the release-set count.

| Operation | RX 570 / renderD129 | RX 580 / renderD128 |
|---|---:|---:|
| Vector add | 2/2 | 2/2 |
| Vector multiply | 15/15 | 15/15 |
| GEMV | 9/9 | 9/9 |
| GEMM | 10/10 | 10/10 |
| Softmax | 16/16 | 16/16 |
| RMSNorm | 16/16 | 16/16 |
| RoPE | 20/20 | 20/20 |
| SiLU | 15/15 | 15/15 |
| Total | 103/103 | 103/103 |

Examples verified their CPU references, guards and fence completion. No
unexpected GPU submission or fence error occurred. Host CTest success is
not used as evidence of hardware execution.

## Real trained-model Turkish CLI rerun

The actual CLI was executed on RX 570 with the source-identical working
backend. Clean-release LLM and converter source files were compared byte for
byte with the working source. The model was the locally converted
Qwen2.5-1.5B-Instruct Q4_K_M checkpoint, stored as FP16 SafeTensors and
expanded to FP32 VRAM. No synthetic weights were used in this run.

```bash
./build-llm/directamd-llm --model /local/qwen2.5-1.5b-fp16 \
  --device /dev/dri/renderD129 --prompt "Türkiye'nin başkenti neresidir?" \
  --context-length 160 --max-tokens 64 --memory-mib 7400
```

Prompt: `Türkiye'nin başkenti neresidir?`

Exact decoded output:

```text
Türkiye'nin başkenti Ankara'dır.
```

- Prompt tokens: 40.
- Generated tokens: 11 including EOS; max-tokens=64 was an upper bound.
- First token including serial prefill, excluding model loading: 117.344821 s.
- Decode throughput: 0.345960 tokens/s.
- Total generation: 146.249905 s.
- Successful AMDGPU submissions: 379127; completed fence waits: 379127.
- Persistent VRAM tensors including weights/cache: 7119142912 bytes.
- Peak tracked tensor allocation including GTT activations: 7120412672 bytes.
- FP16 SafeTensors file: 3554214488 bytes. FP32 raw weight data: 7108352000 bytes.

The earlier development runs produced the same Turkish response on both
cards and are recorded separately. A fresh full Turkish response on RX 580
was not repeated in this publication gate; RX 580 instead reran the real
trained-model layer/logit comparison described below. A full 64-token real
continuation remains unverified; EOS stopped the tested question normally.
Performance is measured, not optimized, and some validation work overlapped
between cards. RX 580 also drives the desktop.

## Real model CPU/GPU reference rerun on RX 580

The optional `llm/validate_trained.py` tool and the clean-release backend
compared two real prompt tokens against independent double-precision CPU
Transformer arithmetic. All 28 layers' critical outputs and logits were
checked: 286 tensor comparisons / 668416 scalar comparisons passed.
Both greedy token selections matched the CPU reference.

Maximum absolute error was 0.0155813453 at layer 26 output, comparing GPU
103.517578125 with CPU 103.501996780. The combined tolerance was
`abs(error) <= 0.002 + 0.002*abs(reference)`. Maximum relative error for
`abs(reference)>1e-4` was 1.24202212; near-zero cancellation makes this
relative statistic large, so absolute and relative tolerances are combined.
This does not establish double-precision equivalence for every generated token.

This check reported 37031 successful submissions and 37031 completed fence
waits, including weight loading, staging and guard checks. Peak tracked tensor
allocation was 7120019456 bytes. The independent CPU/GPU validation process
reached 514984 KiB RSS; this includes bounded mmap reads for the CPU reference.

## Release-set counts and limits

The counted release set is one fresh Turkish CLI run, one fresh real-model
CPU/GPU reference check and the clean release build's 206 GPU regression
cases. It totals **416364 successful AMDGPU submissions and 416364 completed
fence waits**. RX 570 accounts for 379230 and RX 580 for 37134. Counts include
staging and guard-readback kernels. They come from counters incremented only
after successful submit/wait calls; this publication run did not independently
strace every ioctl. Duplicate working-build GPU regressions are excluded.

No model download, training, GPU setting changes or dispatch/GEMV performance
optimization was performed. No failed or unexecuted test is labeled passed.
Model weights, datasets, checkpoints, builds, caches and large/raw test logs
are excluded from the publication. The compact reports and reproducible
source validation tools are included. No existing tracked file is deleted.

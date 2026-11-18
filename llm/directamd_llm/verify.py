"""Independent double-precision Qwen2 reference and local synthetic integration checks."""
import array
import gc
import json
import math
import random
import struct
import tempfile
from pathlib import Path

from .gpu import GPU
from .model import Config, Qwen2
from .weights import SafeWeights, decode_fp32


class Reference:
    def __init__(self, config, weights):
        self.config = config
        self.weights = weights
        self.reset()

    def reset(self):
        self.position = 0
        self.keys = [[] for _ in range(self.config.num_hidden_layers)]
        self.values = [[] for _ in range(self.config.num_hidden_layers)]

    def norm(self, x, weight):
        denominator = math.sqrt(sum(value * value for value in x) / len(x) + self.config.rms_norm_eps)
        return [value * scale / denominator for value, scale in zip(x, weight)]

    @staticmethod
    def project(weight, x):
        return [sum(float(weight[row * len(x) + j]) * x[j] for j in range(len(x)))
                for row in range(len(weight) // len(x))]

    def rope(self, values):
        dim = self.config.head_dim
        output = list(values)
        for head in range(len(values) // dim):
            for pair in range(dim // 2):
                angle = self.position / self.config.rope_theta ** (2 * pair / dim)
                sine, cosine = math.sin(angle), math.cos(angle)
                left = head * dim + pair
                right = left + dim // 2
                output[left] = values[left] * cosine - values[right] * sine
                output[right] = values[left] * sine + values[right] * cosine
        return output

    def step(self, token):
        c, w = self.config, self.weights
        h, dim = c.hidden_size, c.head_dim
        x = list(w["model.embed_tokens.weight"][token * h:(token + 1) * h])
        trace = {"embedding": list(x)}
        for layer in range(c.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            normalized = self.norm(x, w[prefix + "input_layernorm.weight"])
            trace[prefix + "input_norm"] = normalized
            qkv = []
            for name in ("q", "k", "v"):
                stem = prefix + f"self_attn.{name}_proj"
                projection = self.project(w[stem + ".weight"], normalized)
                qkv.append([a + b for a, b in zip(projection, w[stem + ".bias"])])
            q, k, v = qkv
            q, k = self.rope(q), self.rope(k)
            trace[prefix + "q_rope"] = q
            trace[prefix + "v"] = v
            self.keys[layer].append(k)
            self.values[layer].append(v)
            attended = []
            for head in range(c.num_attention_heads):
                kv_head = head // (c.num_attention_heads // c.num_key_value_heads)
                query = q[head * dim:(head + 1) * dim]
                scores = [sum(query[j] * key[kv_head * dim + j] for j in range(dim)) / math.sqrt(dim)
                          for key in self.keys[layer]]
                maximum = max(scores)
                exponentials = [math.exp(value - maximum) for value in scores]
                total = sum(exponentials)
                attended.extend(sum(probability / total * value[kv_head * dim + j]
                                    for probability, value in zip(exponentials, self.values[layer]))
                                for j in range(dim))
            trace[prefix + "attention"] = attended
            residual = self.project(w[prefix + "self_attn.o_proj.weight"], attended)
            x = [a + b for a, b in zip(x, residual)]
            normalized = self.norm(x, w[prefix + "post_attention_layernorm.weight"])
            gate = self.project(w[prefix + "mlp.gate_proj.weight"], normalized)
            up = self.project(w[prefix + "mlp.up_proj.weight"], normalized)
            activated = [a / (1 + math.exp(-a)) * b for a, b in zip(gate, up)]
            residual = self.project(w[prefix + "mlp.down_proj.weight"], activated)
            x = [a + b for a, b in zip(x, residual)]
            trace[prefix + "output"] = list(x)
        normalized = self.norm(x, w["model.norm.weight"])
        trace["final_norm"] = normalized
        head = w["model.embed_tokens.weight"] if c.tie_word_embeddings else w["lm_head.weight"]
        self.position += 1
        return self.project(head, normalized), trace


def synthetic_fixture(directory, config=None):
    # This fixture is deterministic and untrained. It can only validate arithmetic.
    config = config or Config(32, 48, 2, 4, 2, 64, 96)
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen2", **config.__dict__}))
    rng = random.Random(1729)
    header, payload, reference = {}, bytearray(), {}
    for index, (name, shape) in enumerate(config.shapes().items()):
        count = math.prod(shape)
        if name.endswith("layernorm.weight") or name == "model.norm.weight":
            values = [1 + rng.uniform(-0.05, 0.05) for _ in range(count)]
        else:
            values = [rng.uniform(-0.12, 0.12) for _ in range(count)]
        dtype = ("F32", "F16", "BF16")[index % 3]
        fp32 = array.array("f", values)
        if dtype == "F32":
            raw = fp32.tobytes()
        elif dtype == "F16":
            raw = b"".join(struct.pack("<e", value) for value in fp32)
        else:
            raw = b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16)
                           for value in fp32)
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
        reference[name] = decode_fp32(raw, dtype)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return config, reference


def tokenizer_fixture(directory):
    from .tokenizer import Tokenizer
    # Construct a byte-level BPE fixture to check Unicode split, merges and roundtrips.
    keep = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    alphabet, extra = {}, 0
    for value in range(256):
        alphabet[value] = chr(value if value in keep else 256 + extra)
        if value not in keep:
            extra += 1
    vocab = {alphabet[value]: value for value in range(256)}
    vocab["An"] = 256
    vocab["Ank"] = 257
    specials = [{"id": 258, "content": "<|im_start|>", "special": True},
                {"id": 259, "content": "<|im_end|>", "special": True}]
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    raw = {"model": {"type": "BPE", "vocab": vocab, "merges": [["A", "n"], ["An", "k"]]},
           "added_tokens": specials, "normalizer": None, "decoder": {"type": "ByteLevel"},
           "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
               {"type": "Split", "pattern": {"Regex": pattern}, "invert": False},
               {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False}]}}
    (directory / "tokenizer.json").write_text(json.dumps(raw))
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "<|im_start|><|im_end|> add_generation_prompt You are Qwen, created by Alibaba Cloud.",
        "eos_token": "<|im_end|>"}))
    tokenizer = Tokenizer(directory)
    text = "Ankara, Türkiye'nin başkentidir. 123\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode("Ankara")[0] == 257
    suffix = tokenizer.encode("assistant\n")
    assert tokenizer.chat("Türkiye'nin başkenti neresidir?")[-len(suffix):] == suffix
    print("PASS synthetic tokenizer Unicode/BPE/chat roundtrip")


def run(library, device):
    gpu = GPU(library, device, 256 * 1024**2)
    with tempfile.TemporaryDirectory(prefix="directamd-synthetic-") as temp:
        directory = Path(temp)
        config, cpu_weights = synthetic_fixture(directory)
        tokenizer_fixture(directory)
        config = Config.load(directory / "config.json")
        loaded = SafeWeights(directory).load(gpu, config, 80)
        model = Qwen2(gpu, config, loaded, 80)
        reference = Reference(config, cpu_weights)
        worst, comparisons = 0.0, 0
        prompt = [1, 7, 3, 12]
        for index, token in enumerate(prompt + [9, 11]):
            expected, traces = reference.step(token)
            def compare(name, values):
                nonlocal worst, comparisons
                truth = traces[name]
                assert len(values) == len(truth)
                for actual, wanted in zip(values, truth):
                    error = abs(actual - wanted)
                    if not math.isfinite(actual) or error > 2e-5 + 2e-4 * abs(wanted):
                        raise AssertionError(f"{name}: GPU {actual}, CPU {wanted}")
                    worst = max(worst, error)
                comparisons += 1
            actual = model.step(token, compare)
            for value, wanted in zip(actual, expected):
                error = abs(value - wanted)
                assert math.isfinite(value) and error <= 2e-5 + 2e-4 * abs(wanted)
                worst = max(worst, error)
            print(f"PASS synthetic token {index}, cached causal GQA and logits")
        print(f"PASS {comparisons} critical layer comparisons; max absolute error {worst:.9g}")

        # Compare autoregressive greedy token IDs against the independent CPU reference.
        reference.reset()
        for token in prompt:
            logits, _ = reference.step(token)
        expected_ids = []
        for index in range(64):
            token = max(range(len(logits)), key=logits.__getitem__)
            expected_ids.append(token)
            if index + 1 < 64:
                logits, _ = reference.step(token)
        actual_ids, timing = model.generate(prompt, 64, set())
        assert actual_ids == expected_ids, (actual_ids, expected_ids)
        print("PASS synthetic 64-token autoregressive CPU/GPU ID match")
        print("Synthetic token IDs:", actual_ids)
        print("Synthetic timing:", json.dumps(timing, sort_keys=True))
        eos_ids, _ = model.generate(prompt, 64, {expected_ids[0]})
        assert eos_ids == expected_ids[:1]
        print("PASS EOS stopping and request/cache reset")

        model.reset()
        replay = [model.step(token) for token in prompt]
        model.reset()
        assert [list(model.step(token)) for token in prompt] == [list(x) for x in replay]
        print("PASS cache replay deterministic")
        model.position = model.capacity
        try:
            model.step(1)
        except ValueError:
            pass
        else:
            raise AssertionError("KV overflow accepted")
        try:
            model.step(config.vocab_size)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid token accepted")
        tensor = gpu.tensor(4)
        try:
            tensor.view(3, 2)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid tensor view accepted")
        print("PASS context/token/tensor bounds and GPU guards")
        # Exercise the real target's head width and GQA ratio with a small vocabulary.
        del model, reference, loaded, tensor
        gc.collect()
        wide_config = Config(896, 128, 1, 14, 2, 128, 16)
        wide_config, wide_cpu = synthetic_fixture(directory, wide_config)
        wide_weights = SafeWeights(directory).load(gpu, wide_config, 8)
        wide_model = Qwen2(gpu, wide_config, wide_weights, 8)
        wide_ref = Reference(wide_config, wide_cpu)
        wide_worst = 0.0
        for token in (2, 5, 11):
            expected, traces = wide_ref.step(token)
            def compare_wide(name, values):
                nonlocal wide_worst
                for actual, wanted in zip(values, traces[name]):
                    error = abs(actual - wanted)
                    if not math.isfinite(actual) or error > 3e-5 + 3e-4 * abs(wanted):
                        raise AssertionError(f"Qwen-width {name}: GPU {actual}, CPU {wanted}")
                    wide_worst = max(wide_worst, error)
            actual = wide_model.step(token, compare_wide)
            for value, wanted in zip(actual, expected):
                assert abs(value - wanted) < 3e-5 + 3e-4 * abs(wanted)
                wide_worst = max(wide_worst, abs(value - wanted))
        print(f"PASS synthetic Qwen head geometry H=896, Q=14, KV=2, D=64; max abs error {wide_worst:.9g}")

        # Reject malformed files before any tensor upload.
        malformed = directory / "malformed"
        malformed.mkdir()
        encoded = json.dumps({"bad": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()
        (malformed / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
        try:
            SafeWeights(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("Truncated SafeTensors payload accepted")
        print("PASS SafeTensors truncated payload rejection")

        stats = gpu.stats()
        assert stats["submissions"] == stats["fences"] and stats["submissions"] > 0
        print("GPU statistics:", json.dumps(stats, sort_keys=True))
        print("REAL TRAINED LANGUAGE GENERATION: UNVERIFIED (synthetic weights only)")

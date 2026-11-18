"""Qwen2 FP32 forward pass, with persistent weights and causal GQA KV cache."""
import array
import json
import math
import time
from dataclasses import dataclass


@dataclass
class Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

    @classmethod
    def load(cls, path):
        raw = json.loads(path.read_text())
        if raw.get("model_type") != "qwen2" or raw.get("hidden_act", "silu") != "silu":
            raise ValueError("Only Qwen2 with SiLU activation is supported")
        if raw.get("use_sliding_window") or raw.get("rope_scaling"):
            raise ValueError("Sliding-window attention and scaled RoPE require a separate implementation")
        config = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        config.validate()
        if raw.get("head_dim", config.head_dim) != config.head_dim:
            raise ValueError("Custom head_dim is unsupported")
        return config

    def validate(self):
        for name in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                     "num_key_value_heads", "vocab_size", "max_position_embeddings"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Invalid positive integer in model config: {name}")
        if (self.hidden_size % self.num_attention_heads or
                self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2):
            raise ValueError("Invalid Qwen2 GQA/head dimensions")
        if type(self.tie_word_embeddings) is not bool:
            raise ValueError("tie_word_embeddings must be boolean")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("Invalid RMSNorm epsilon")
        if not math.isfinite(self.rope_theta) or self.rope_theta < 1:
            raise ValueError("Invalid RoPE base")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def shapes(self):
        h, m, v = self.hidden_size, self.intermediate_size, self.vocab_size
        kv = self.num_key_value_heads * self.head_dim
        shapes = {"model.embed_tokens.weight": (v, h), "model.norm.weight": (h,)}
        for layer in range(self.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            shapes[prefix + "input_layernorm.weight"] = (h,)
            shapes[prefix + "post_attention_layernorm.weight"] = (h,)
            for name, rows in (("q", h), ("k", kv), ("v", kv)):
                shapes[prefix + f"self_attn.{name}_proj.weight"] = (rows, h)
                shapes[prefix + f"self_attn.{name}_proj.bias"] = (rows,)
            shapes[prefix + "self_attn.o_proj.weight"] = (h, h)
            for name in ("gate", "up"):
                shapes[prefix + f"mlp.{name}_proj.weight"] = (m, h)
            shapes[prefix + "mlp.down_proj.weight"] = (h, m)
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (v, h)
        return shapes


class Qwen2:
    def __init__(self, gpu, config, weights, context_length):
        config.validate()
        if context_length < 1 or context_length > config.max_position_embeddings:
            raise ValueError("Context length outside model limits")
        self.gpu, self.config, self.weights = gpu, config, weights
        self.capacity = context_length
        self.position = 0
        d = config.head_dim
        self.keys = [[gpu.tensor(context_length * d, vram=True) for _ in range(config.num_key_value_heads)]
                     for _ in range(config.num_hidden_layers)]
        self.values = [[gpu.tensor(context_length * d, vram=True) for _ in range(config.num_key_value_heads)]
                       for _ in range(config.num_hidden_layers)]
        self.scale = {}
        self.phase = {}

    def reset(self):
        # Future attention views expose only the cache prefix rewritten by this request.
        self.position = 0

    def step(self, token, trace=None):
        c, gpu, weights = self.config, self.gpu, self.weights
        if not 0 <= token < c.vocab_size:
            raise ValueError("Token ID outside vocabulary")
        if self.position >= self.capacity:
            raise ValueError("KV cache capacity exceeded")
        position, dim = self.position, c.head_dim
        if position not in self.phase:
            turns = [math.remainder(position * c.rope_theta ** (-2 * pair / dim), 2 * math.pi) /
                     (2 * math.pi) for pair in range(dim // 2)]
            self.phase[position] = gpu.tensor(len(turns), turns)
        phase = self.phase[position]
        # Embedding gather copies the selected row through the GPU copy/add path.
        x = gpu.copy(weights["model.embed_tokens.weight"].view(token * c.hidden_size, c.hidden_size))
        if trace:
            trace("embedding", x.read())
        for layer in range(c.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            normalized = gpu.norm(x, weights[prefix + "input_layernorm.weight"], c.rms_norm_eps)
            if trace:
                trace(prefix + "input_norm", normalized.read())
            projections = []
            for name, rows in (("q", c.hidden_size),
                               ("k", c.num_key_value_heads * dim), ("v", c.num_key_value_heads * dim)):
                stem = prefix + f"self_attn.{name}_proj"
                value = gpu.gemv(weights[stem + ".weight"], normalized, rows)
                projections.append(gpu.add(value, weights[stem + ".bias"]))
            q, k, v = projections
            q_heads = [gpu.rope(q.view(head * dim, dim), phase) for head in range(c.num_attention_heads)]
            k_heads = [gpu.rope(k.view(head * dim, dim), phase) for head in range(c.num_key_value_heads)]
            if trace:
                trace(prefix + "q_rope", [value for head in q_heads for value in head.read()])
                trace(prefix + "v", v.read())
            for head in range(c.num_key_value_heads):
                gpu.copy(k_heads[head], self.keys[layer][head].view(position * dim, dim))
                gpu.copy(v.view(head * dim, dim), self.values[layer][head].view(position * dim, dim))
            length = position + 1
            if length not in self.scale:
                self.scale[length] = gpu.tensor(length, [1.0 / math.sqrt(dim)] * length)
            attention = gpu.tensor(c.hidden_size)
            group = c.num_attention_heads // c.num_key_value_heads
            for head in range(c.num_attention_heads):
                kv_head = head // group
                key = self.keys[layer][kv_head].view(0, length * dim)
                value = self.values[layer][kv_head].view(0, length * dim)
                scores = gpu.gemv(key, q_heads[head], length)
                probabilities = gpu.softmax(gpu.multiply(scores, self.scale[length]))
                # [1, length] @ [length, head_dim]; only the causal cache prefix is exposed.
                attended = gpu.gemm(probabilities, value, 1, dim, length)
                gpu.copy(attended, attention.view(head * dim, dim))
            if trace:
                trace(prefix + "attention", attention.read())
            projected = gpu.gemv(weights[prefix + "self_attn.o_proj.weight"], attention, c.hidden_size)
            x = gpu.add(x, projected)
            normalized = gpu.norm(x, weights[prefix + "post_attention_layernorm.weight"], c.rms_norm_eps)
            gate = gpu.gemv(weights[prefix + "mlp.gate_proj.weight"], normalized, c.intermediate_size)
            up = gpu.gemv(weights[prefix + "mlp.up_proj.weight"], normalized, c.intermediate_size)
            activated = gpu.multiply(gpu.silu(gate), up)
            x = gpu.add(x, gpu.gemv(weights[prefix + "mlp.down_proj.weight"], activated, c.hidden_size))
            if trace:
                trace(prefix + "output", x.read())
        normalized = gpu.norm(x, weights["model.norm.weight"], c.rms_norm_eps)
        head = weights["model.embed_tokens.weight"] if c.tie_word_embeddings else weights["lm_head.weight"]
        logits = gpu.gemv(head, normalized, c.vocab_size)
        if trace:
            trace("final_norm", normalized.read())
        # The logical cache advances only after every layer and fence has succeeded.
        self.position += 1
        return logits.read()

    def generate(self, prompt_ids, maximum, eos_ids, on_token=None):
        if not prompt_ids or maximum < 1 or len(prompt_ids) + maximum > self.capacity:
            raise ValueError("Prompt plus requested generation exceeds the KV cache budget")
        self.reset()
        start = time.perf_counter()
        logits = None
        for token in prompt_ids:
            logits = self.step(token)
        first = time.perf_counter() - start
        generated = []
        decode_start = time.perf_counter()
        for index in range(maximum):
            if not all(math.isfinite(value) for value in logits):
                raise RuntimeError("Nonfinite model logits")
            token = max(range(len(logits)), key=logits.__getitem__)
            generated.append(token)
            if on_token:
                on_token(token)
            if token in eos_ids:
                break
            if index + 1 < maximum:
                logits = self.step(token)
        elapsed = time.perf_counter() - decode_start
        return generated, {"first_token_seconds": first,
                           "decode_tokens_per_second": (len(generated) - 1) / elapsed if elapsed else 0,
                           "generated_tokens": len(generated), "total_seconds": time.perf_counter() - start}

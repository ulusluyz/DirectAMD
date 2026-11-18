"""Bounded-memory GGUF v2/v3 Qwen2 to FP16 SafeTensors conversion, local files only."""
import argparse, ctypes as ct, json, math, os, struct, sys, time, resource, shutil
from pathlib import Path
from .gguf import GGUF
from .model import Config
from .weights import SafeWeights
TYPES = {0: (1, 4, 'F32'), 1: (1, 2, 'F16'), 12: (256, 144, 'Q4_K'), 14: (256, 210, 'Q6_K')}

def destination(name):
    simple = {'token_embd.weight': 'model.embed_tokens.weight', 'output_norm.weight': 'model.norm.weight', 'output.weight': 'lm_head.weight'}
    if name in simple:
        return simple[name]
    p = name.split('.')
    maps = {'attn_norm': 'input_layernorm', 'ffn_norm': 'post_attention_layernorm', 'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj', 'attn_v': 'self_attn.v_proj', 'attn_output': 'self_attn.o_proj', 'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj', 'ffn_down': 'mlp.down_proj'}
    if len(p) != 4 or p[0] != 'blk' or (not p[1].isdigit()) or (p[2] not in maps) or (p[3] not in ('weight', 'bias')):
        raise ValueError(f'Unsupported tensor name {name}')
    return f'model.layers.{p[1]}.{maps[p[2]]}.{p[3]}'

def config(g):
    m = g.meta
    if m.get('general.architecture') != 'qwen2':
        raise ValueError('Only qwen2 supported')
    keys = {'hidden_size': 'embedding_length', 'intermediate_size': 'feed_forward_length', 'num_hidden_layers': 'block_count', 'num_attention_heads': 'attention.head_count', 'num_key_value_heads': 'attention.head_count_kv', 'max_position_embeddings': 'context_length', 'rms_norm_eps': 'attention.layer_norm_rms_epsilon', 'rope_theta': 'rope.freq_base'}
    c = Config(**{k: m['qwen2.' + v] for k, v in keys.items()}, vocab_size=len(m['tokenizer.ggml.tokens']), tie_word_embeddings=not any((t['name'] == 'output.weight' for t in g.tensors)))
    c.validate()
    if any((m.get('qwen2.attention.' + k, c.head_dim) != c.head_dim for k in ('key_length', 'value_length'))):
        raise ValueError('Custom attention head dimensions unsupported')
    if m.get('qwen2.attention.sliding_window', 0):
        raise ValueError('Sliding-window attention unsupported')
    if m.get('qwen2.rope.dimension_count', c.head_dim) != c.head_dim:
        raise ValueError('Partial rotary dimensions unsupported')
    if any(('rope.scaling' in k for k in m)):
        raise ValueError('Scaled RoPE unsupported')
    return c
PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

def tokenizer(g, out):
    m = g.meta
    if m.get('tokenizer.ggml.model') != 'gpt2' or m.get('tokenizer.ggml.pre') != 'qwen2' or m.get('tokenizer.ggml.add_bos_token', False):
        raise ValueError('Unsupported GGUF tokenizer')
    tokens = m['tokenizer.ggml.tokens']
    types = m['tokenizer.ggml.token_type']
    if len(tokens) != len(types) or len(set(tokens)) != len(tokens):
        raise ValueError('Invalid vocabulary')
    if set(types) - {1, 3, 4, 5}:
        raise ValueError(f'Unsupported token types {set(types)}')
    special = [dict(id=i, content=t, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True) for i, (t, typ) in enumerate(zip(tokens, types)) if typ in (3, 4)]
    obj = dict(version='1.0', truncation=None, padding=None, added_tokens=special, normalizer=None, pre_tokenizer=dict(type='Sequence', pretokenizers=[dict(type='Split', pattern={'Regex': PATTERN}, behavior='Isolated', invert=False), dict(type='ByteLevel', add_prefix_space=False, trim_offsets=True, use_regex=False)]), decoder=dict(type='ByteLevel', add_prefix_space=False, trim_offsets=True, use_regex=False), post_processor=None, model=dict(type='BPE', dropout=None, unk_token=None, continuing_subword_prefix='', end_of_word_suffix='', fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab={t: i for i, t in enumerate(tokens)}, merges=m['tokenizer.ggml.merges']))
    write(out / 'tokenizer.json', obj)
    write(out / 'tokenizer_config.json', dict(tokenizer_class='Qwen2Tokenizer', eos_token=tokens[m['tokenizer.ggml.eos_token_id']], pad_token=tokens[m['tokenizer.ggml.padding_token_id']], bos_token=None, chat_template=m['tokenizer.chat_template'], model_max_length=m['qwen2.context_length']))
    write(out / 'generation_config.json', dict(eos_token_id=[m['tokenizer.ggml.eos_token_id'], m['tokenizer.ggml.bos_token_id']], pad_token_id=m['tokenizer.ggml.padding_token_id']))

def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def plan(g, c):
    result = []
    names = set()
    spans = []
    for t in g.tensors:
        if t['type'] not in TYPES:
            raise ValueError(f"Unsupported type {t['type']} in {t['name']}")
        name = destination(t['name'])
        shape = tuple(reversed(t['dims']))
        count = math.prod(shape)
        block, width, _ = TYPES[t['type']]
        if any((d < 1 for d in shape)) or count > (1 << 30) - 128 or count % block or t['dims'][0] % block:
            raise ValueError('Misaligned quantized tensor')
        if name in names or c.shapes().get(name) != shape:
            raise ValueError(f'Wrong shape or duplicate tensor {name} {shape}')
        names.add(name)
        start = g.start + t['offset']
        end = start + count // block * width
        if t['offset'] % g.meta.get('general.alignment', 32) or start < g.start or end > g.path.stat().st_size:
            raise ValueError('Tensor outside GGUF file')
        spans.append((start, end))
        result.append((t, name, shape, count))
    if names != set(c.shapes()):
        raise ValueError(f'Missing tensors {set(c.shapes()) - names}')
    spans.sort()
    if any((a[1] > b[0] for a, b in zip(spans, spans[1:]))):
        raise ValueError('Overlapping tensors')
    return result

def main(library):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--inspect', action='store_true')
    args = p.parse_args()
    try:
        g = GGUF(args.input)
        c = config(g)
        items = plan(g, c)
        report = dict(unused_vocabulary_rows=g.meta['tokenizer.ggml.token_type'].count(5), gguf_version=g.version, source_bytes=g.path.stat().st_size, tensor_count=len(items), types={TYPES[t][2]: sum((x[0]['type'] == t for x in items)) for t in set((x[0]['type'] for x in items))}, config=c.__dict__, fp16_data_bytes=sum((x[3] * 2 for x in items)), fp32_weight_bytes=sum((x[3] * 4 for x in items)))
        if args.inspect:
            print(json.dumps(report, indent=2))
            return 0
        if not args.output:
            raise ValueError('--output required for conversion')
        out = args.output.resolve()
        if out.exists():
            raise ValueError('Output directory must not already exist; refusing overwrite')
        parent = out.parent
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < report['fp16_data_bytes'] + 64 * 1024 ** 2:
            raise ValueError('Insufficient disk space for FP16 conversion')
        out.mkdir(parents=True)
        (out / 'INCOMPLETE').write_text('Conversion not yet validated\n')
        lib = ct.CDLL(library)
        lib.decode.argtypes = [ct.c_int, ct.c_void_p, ct.c_uint64, ct.c_void_p]
        lib.decode.restype = ct.c_int
        begin = time.monotonic()
        header = {'__metadata__': {'format': 'pt', 'source': 'GGUF dequantization; original quantization loss is irreversible'}}
        offset = 0
        for t, name, shape, count in items:
            header[name] = dict(dtype='F16', shape=shape, data_offsets=[offset, offset + count * 2])
            offset += count * 2
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' ' * (-len(raw) % 8)
        with (out / 'model.safetensors').open('xb') as f:
            f.write(struct.pack('<Q', len(raw)))
            f.write(raw)
            for t, name, shape, count in items:
                g.f.seek(g.start + t['offset'])
                block, width, _ = TYPES[t['type']]
                for i in range(0, count, 262144):
                    n = min(262144, count - i)
                    data = g.f.read(n // block * width)
                    if len(data) != n // block * width:
                        raise ValueError('Truncated tensor')
                    output = ct.create_string_buffer(n * 2)
                    if lib.decode(t['type'], data, n, output):
                        raise ValueError(f'Nonfinite/overflow or unsupported tensor {name}')
                    f.write(output.raw)
                print(name, shape, flush=True)
        write(out / 'config.json', dict(c.__dict__, model_type='qwen2', architectures=['Qwen2ForCausalLM'], hidden_act='silu', torch_dtype='float16', use_sliding_window=False))
        tokenizer(g, out)
        loaded = SafeWeights(out, allow_incomplete=True)
        if len(loaded.entries) != len(items) or any((loaded.entries[n][2:] != ('F16', s) for _, n, s, _ in items)):
            raise ValueError('Output validation failed')
        report.update(output_bytes=(out / 'model.safetensors').stat().st_size, seconds=time.monotonic() - begin, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(out / 'conversion-report.json', report)
        (out / 'INCOMPLETE').unlink()
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, KeyError, OverflowError) as e:
        print(f'DirectAMD conversion error: {e}', file=sys.stderr)
        return 1

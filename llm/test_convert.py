"""CPU converter layout, bounds, unsupported-type and numeric regression tests."""
import ctypes as ct, json, struct, tempfile, unittest, sys, math, subprocess, hashlib
from directamd_llm.model import Config
from directamd_llm.weights import SafeWeights
from directamd_llm.verify import tokenizer_fixture
from pathlib import Path
from directamd_llm.gguf import GGUF
from directamd_llm.convert import destination, plan
LIB = ct.CDLL(sys.argv.pop())
LIB.decode.argtypes = [ct.c_int, ct.c_void_p, ct.c_uint64, ct.c_void_p]

def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b

class Converter(unittest.TestCase):

    def decode(self, t, raw, n):
        out = ct.create_string_buffer(n * 2)
        self.assertEqual(LIB.decode(t, raw, n, out), 0)
        return struct.unpack('<' + 'e' * n, out.raw)

    def test_q4_scale_layout(self):
        raw = struct.pack('<ee', 0.5, 0.25) + bytes([1, 2, 3, 4, 5, 6, 7, 8, 145, 162, 179, 196]) + bytes([196] * 128)
        out = self.decode(12, raw, 256)
        for group in range(8):
            sc = [1, 2, 3, 4, 1, 2, 3, 4][group]
            mn = [5, 6, 7, 8, 9, 10, 11, 12][group]
            self.assertEqual(out[group * 32:(group + 1) * 32], tuple([0.5 * sc * (4 if group % 2 == 0 else 12) - 0.25 * mn] * 32))

    def test_q6_signed_scales_and_order(self):
        q = [i % 64 - 32 for i in range(256)]
        sc = [(-1 if i % 2 else 1) * (i + 1) for i in range(16)]
        lo = bytearray(128)
        hi = bytearray(64)
        for n in range(2):
            for l in range(32):
                vals = [q[n * 128 + l + s * 32] + 32 for s in range(4)]
                lo[n * 64 + l] = vals[0] & 15 | (vals[2] & 15) << 4
                lo[n * 64 + l + 32] = vals[1] & 15 | (vals[3] & 15) << 4
                hi[n * 32 + l] = sum((v >> 4 << 2 * s for s, v in enumerate(vals)))
        raw = lo + hi + struct.pack('<16b', *sc) + struct.pack('<e', 0.125)
        self.assertEqual(self.decode(14, bytes(raw), 256), tuple((0.125 * sc[i // 16] * q[i] for i in range(256))))

    def test_nonfinite_overflow_alignment(self):
        out = ct.create_string_buffer(512)
        for v in [float('inf'), float('nan'), 1e+20]:
            self.assertNotEqual(LIB.decode(0, struct.pack('<f', v), 1, out), 0)
        self.assertNotEqual(LIB.decode(99, b'\x00' * 256, 256, out), 0)
        self.assertNotEqual(LIB.decode(12, b'\x00' * 144, 255, out), 0)

    def test_complete_conversion_and_incomplete_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            tokenizer_fixture(directory)
            tokenizer = json.loads((directory / 'tokenizer.json').read_text())
            vocab = [token for token, index in sorted(tokenizer['model']['vocab'].items(), key=lambda item: item[1])]
            vocab += ['<|im_start|>', '<|im_end|>']
            c = Config(4, 8, 1, 2, 1, len(vocab), 32)
            meta = {'general.architecture': 'qwen2', 'qwen2.embedding_length': 4,
                    'qwen2.feed_forward_length': 8, 'qwen2.block_count': 1,
                    'qwen2.attention.head_count': 2, 'qwen2.attention.head_count_kv': 1,
                    'qwen2.context_length': 32, 'qwen2.attention.layer_norm_rms_epsilon': 1e-6,
                    'qwen2.rope.freq_base': 1e6, 'tokenizer.ggml.model': 'gpt2',
                    'tokenizer.ggml.pre': 'qwen2', 'tokenizer.ggml.tokens': vocab,
                    'tokenizer.ggml.token_type': [1] * (len(vocab)-2) + [3, 3],
                    'tokenizer.ggml.merges': ['A n', 'An k'],
                    'tokenizer.ggml.eos_token_id': len(vocab)-1,
                    'tokenizer.ggml.bos_token_id': len(vocab)-1,
                    'tokenizer.ggml.padding_token_id': len(vocab)-1,
                    'tokenizer.chat_template': json.loads((directory / 'tokenizer_config.json').read_text())['chat_template']}
            def value(v):
                if isinstance(v, str): return 8, string(v)
                if isinstance(v, int): return 4, struct.pack('<I', v)
                if isinstance(v, float): return 6, struct.pack('<f', v)
                t = 8 if v and isinstance(v[0], str) else 4
                return 9, struct.pack('<IQ', t, len(v)) + b''.join(value(x)[1] for x in v)
            metadata = b''
            for key, v in meta.items():
                typ, raw = value(v)
                metadata += string(key) + struct.pack('<I', typ) + raw
            mapping = {'token_embd.weight':'model.embed_tokens.weight', 'output_norm.weight':'model.norm.weight'}
            for part in ['attn_norm', 'ffn_norm', 'attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down']:
                for suffix in ['weight', 'bias']:
                    name = 'blk.0.' + part + '.' + suffix
                    mapping[name] = destination(name)
            inverse = {v:k for k,v in mapping.items()}
            table = payload = b''
            for name, shape in c.shapes().items():
                payload += b'\0' * (-len(payload) % 32)
                table += string(inverse[name]) + struct.pack('<I', len(shape))
                table += struct.pack('<'+'Q'*len(shape), *reversed(shape)) + struct.pack('<IQ', 0, len(payload))
                payload += struct.pack('<f', 0.03125) * math.prod(shape)
            header = b'GGUF' + struct.pack('<IQQ', 3, len(c.shapes()), len(meta)) + metadata + table
            source = directory / 'model.gguf'
            source.write_bytes(header + b'\0'*(-len(header)%32) + payload)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target = directory / 'converted'
            cli = Path(LIB._name).resolve().parent.parent / 'directamd-convert'
            result = subprocess.run([str(cli), '--input', str(source), '--output', str(target)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((target/'INCOMPLETE').exists())
            self.assertEqual(set(SafeWeights(target).entries), set(c.shapes()))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            (target/'INCOMPLETE').write_text('test')
            with self.assertRaises(ValueError): SafeWeights(target)

    def test_fp16_values(self):
        raw = struct.pack('<6e', -0.0, 0.0, 2**-24, -2**-14, 65504, -65504)
        out = ct.create_string_buffer(len(raw))
        self.assertEqual(LIB.decode(1, raw, 6, out), 0)
        self.assertEqual(out.raw, raw)

    def test_names(self):
        self.assertEqual(destination('blk.27.attn_q.weight'), 'model.layers.27.self_attn.q_proj.weight')
        self.assertEqual(destination('output.weight'), 'lm_head.weight')
        with self.assertRaises(ValueError):
            destination('unknown.weight')

    def test_gguf_bounds_and_duplicate_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.gguf'
            for raw in [b'GGUF' + struct.pack('<IQQ', 1, 0, 0), b'GGUF' + struct.pack('<IQQ', 3, 100001, 0), b'GGUF' + struct.pack('<IQQ', 3, 0, 1) + struct.pack('<Q', 99999999), b'GGUF' + struct.pack('<IQQ', 3, 0, 2) + (string('key') + struct.pack('<II', 4, 1)) * 2]:
                p.write_bytes(raw)
                with self.assertRaises(ValueError):
                    GGUF(p)

    def test_tensor_extent_and_types(self):

        class Config:

            def shapes(self):
                return {'model.norm.weight': (1,)}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x'
            p.write_bytes(b'\x00' * 64)

            class File:
                pass
            g = File()
            g.path = p
            g.start = 32
            g.meta = {}
            g.tensors = [dict(name='output_norm.weight', dims=[1], type=0, offset=0)]
            self.assertEqual(len(plan(g, Config())), 1)
            for changes in [dict(type=13), dict(offset=64), dict(offset=1), dict(dims=[0])]:
                g.tensors = [dict(name='output_norm.weight', dims=[1], type=0, offset=0, **{})]
                g.tensors[0].update(changes)
                with self.assertRaises(ValueError):
                    plan(g, Config())
if __name__ == '__main__':
    unittest.main()

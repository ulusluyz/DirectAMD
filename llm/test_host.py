"""Local-only loader, model-config and tokenizer regression tests."""
import array
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from directamd_llm.model import Config
from directamd_llm.verify import synthetic_fixture, tokenizer_fixture
from directamd_llm.weights import SafeWeights, decode_fp32


class ModelFiles(unittest.TestCase):
    def test_local_mixed_dtype_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config, reference = synthetic_fixture(directory)
            loaded = Config.load(directory / "config.json")
            self.assertEqual(loaded, config)
            files = SafeWeights(directory)
            self.assertEqual(set(files.entries), set(config.shapes()))
            for name, (_, _, _, shape) in files.entries.items():
                self.assertEqual(math.prod(shape), len(reference[name]))

    def test_conversion(self):
        for dtype, raw in (("F32", struct.pack("<3f", 1, -2, 0.5)),
                           ("F16", struct.pack("<3e", 1, -2, 0.5)),
                           ("BF16", struct.pack("<3H", 0x3f80, 0xc000, 0x3f00))):
            self.assertEqual(list(decode_fp32(raw, dtype)), [1, -2, 0.5])

    def test_invalid_config(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            config = Config(32, 48, 2, 4, 2, 64, 96).__dict__
            for changes in ({"hidden_size": 33}, {"rms_norm_eps": 0}, {"num_key_value_heads": 3},
                            {"rope_scaling": {"type": "dynamic"}}, {"use_sliding_window": True}):
                path.write_text(json.dumps({"model_type": "qwen2", **config, **changes}))
                with self.assertRaises(ValueError):
                    Config.load(path)

    def test_bad_tensor_ranges(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            for descriptor, payload in (
                ({"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}, b""),
                ({"dtype": "F16", "shape": [4], "data_offsets": [-1, 7]}, b"\0" * 8),
                ({"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}, b"\0" * 4),
            ):
                raw = json.dumps({"weight": descriptor}).encode()
                path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
                with self.assertRaises(ValueError):
                    SafeWeights(Path(temp))

    def test_tokenizer_unicode_and_chat(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer_fixture(Path(temp))


if __name__ == "__main__":
    unittest.main()

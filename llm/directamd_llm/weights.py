"""Local-only SafeTensors reader with checked offsets and FP32 conversion."""
import array
import json
import math
import struct
import sys
from pathlib import Path


class SafeWeights:
    def __init__(self, directory, *, allow_incomplete=False):
        self.directory = Path(directory).resolve()
        if not allow_incomplete and (self.directory / "INCOMPLETE").exists():
            raise ValueError("Incomplete model conversion; refusing to load")
        index = self.directory / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text()).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Invalid SafeTensors index")
            files = sorted(set(mapping.values()))
        else:
            mapping = None
            files = [path.name for path in self.directory.glob("*.safetensors")]
        if not files:
            raise FileNotFoundError("No local SafeTensors weights found")
        self.entries = {}
        for name in files:
            path = (self.directory / name).resolve()
            if not path.is_relative_to(self.directory) or not path.is_file():
                raise ValueError("Invalid SafeTensors shard path")
            size = path.stat().st_size
            with path.open("rb") as stream:
                raw = stream.read(8)
                if len(raw) != 8:
                    raise ValueError("Truncated SafeTensors file")
                header_size = struct.unpack("<Q", raw)[0]
                if not 2 <= header_size <= min(64 * 1024**2, size - 8):
                    raise ValueError("Invalid SafeTensors header length")
                header = json.loads(stream.read(header_size))
            if not isinstance(header, dict):
                raise ValueError("Invalid SafeTensors header")
            data_start = 8 + header_size
            spans = []
            for tensor_name, item in header.items():
                if tensor_name == "__metadata__":
                    continue
                if tensor_name in self.entries:
                    raise ValueError("Duplicate tensor in SafeTensors shards")
                dtype = item.get("dtype")
                shape = item.get("shape")
                offsets = item.get("data_offsets")
                if dtype not in ("F32", "F16", "BF16"):
                    raise ValueError(f"Unsupported SafeTensors dtype {dtype}")
                if not isinstance(shape, list) or not shape or any(type(n) is not int or n < 1 for n in shape):
                    raise ValueError("Invalid tensor dimensions")
                if (not isinstance(offsets, list) or len(offsets) != 2 or
                        any(type(n) is not int for n in offsets)):
                    raise ValueError("Invalid tensor data offsets")
                begin, end = offsets
                count = math.prod(shape)
                width = 4 if dtype == "F32" else 2
                if count > (1 << 30) - 128 or begin < 0 or end < begin or end - begin != count * width:
                    raise ValueError("Invalid tensor extent")
                if end > size - data_start:
                    raise ValueError("SafeTensors data exceeds shard size")
                spans.append((begin, end))
                self.entries[tensor_name] = (path, data_start + begin, dtype, tuple(shape))
            spans.sort()
            if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
                raise ValueError("Overlapping SafeTensors tensor spans")
        if mapping is not None:
            for name, shard in mapping.items():
                if name not in self.entries or self.entries[name][0] != (self.directory / shard).resolve():
                    raise ValueError("SafeTensors index does not match shard contents")

    def load(self, gpu, config, context_length):
        shapes = config.shapes()
        for name, shape in shapes.items():
            if name not in self.entries or self.entries[name][3] != shape:
                raise ValueError(f"Missing or wrong-shaped tensor: {name}, expected {shape}")
        weight_bytes = sum(((math.prod(shape) + 128) * 4 + 4095) // 4096 * 4096
                           for shape in shapes.values())
        cache_allocation = ((context_length * config.head_dim + 128) * 4 + 4095) // 4096 * 4096
        cache_bytes = config.num_hidden_layers * config.num_key_value_heads * 2 * cache_allocation
        # Reserve explicit working space beyond the persistent weights and KV buffers.
        reserve = max(64 * 1024**2, config.intermediate_size * 128)
        stats = gpu.stats()
        if weight_bytes + cache_bytes + reserve > stats["budget_bytes"] - stats["resident_bytes"]:
            raise ValueError("FP32 weights, KV cache and scratch space exceed the GPU memory budget")
        result = {}
        for name, shape in shapes.items():
            path, offset, dtype, _ = self.entries[name]
            count = math.prod(shape)
            tensor = gpu.tensor(count, vram=True)
            width = 4 if dtype == "F32" else 2
            with path.open("rb") as stream:
                stream.seek(offset)
                for start in range(0, count, 262144):
                    length = min(262144, count - start)
                    raw = stream.read(length * width)
                    if len(raw) != length * width:
                        raise ValueError("Truncated tensor data")
                    if dtype == "F16":
                        tensor.view(start, length).upload_fp16(raw)
                        continue
                    values = decode_fp32(raw, dtype)
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(f"Nonfinite weights in {name}")
                    tensor.view(start, length).upload(values)
            result[name] = tensor
        return result


def decode_fp32(raw, dtype):
    if dtype == "F32":
        values = array.array("f")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return values
    if dtype == "F16":
        return array.array("f", (value[0] for value in struct.iter_unpack("<e", raw)))
    if dtype == "BF16":
        bits = array.array("I", (value[0] << 16 for value in struct.iter_unpack("<H", raw)))
        values = array.array("f")
        values.frombytes(bits.tobytes())
        return values
    raise ValueError("Unsupported tensor dtype")

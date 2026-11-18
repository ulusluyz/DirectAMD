"""Optional FP16 model check using an independent double-precision CPU reference."""
import sys, ctypes as ct, mmap, struct, math, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from directamd_llm.weights import SafeWeights
from directamd_llm.model import Config
from directamd_llm.verify import Reference
root = None
reference_library = None

class HalfTensor:

    def __init__(self, entry):
        self.path, self.offset, self.dtype, self.shape = entry
        self.count = math.prod(self.shape)
        if self.dtype != "F16":
            raise ValueError("This reference check requires FP16 weights")

    def __len__(self):
        return self.count

    def __getitem__(self, key):
        start, stop, step = key.indices(self.count) if isinstance(key, slice) else (key, key + 1, 1)
        if step != 1:
            raise ValueError('slice')
        with self.path.open('rb') as f:
            f.seek(self.offset + 2 * start)
            raw = f.read(2 * (stop - start))
        values = [x[0] for x in struct.iter_unpack('<e', raw)]
        return values if isinstance(key, slice) else values[0]

    def __iter__(self):
        return iter(self[:])

class StreamingReference(Reference):

    def __init__(self, c):
        self.lib = ct.CDLL(str(reference_library))
        self.lib.reference_gemv.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint64, ct.c_uint64]
        super().__init__(c, {n: HalfTensor(v) for n, v in SafeWeights(root).entries.items()})

    def project(self, w, x):
        rows = len(w) // len(x)
        a = (ct.c_double * len(x))(*x)
        b = (ct.c_double * rows)()
        with w.path.open('rb') as f:
            page = w.offset // mmap.PAGESIZE * mmap.PAGESIZE
            shift = w.offset - page
            with mmap.mmap(f.fileno(), shift + w.count * 2, access=mmap.ACCESS_COPY, offset=page) as mapped:
                ptr = ct.c_char.from_buffer(mapped)
                address = ct.addressof(ptr) + shift
                del ptr
                self.lib.reference_gemv(address, a, b, rows, len(x))
        return list(b)

def compare(model, references):
    records = references
    worst = relative = 0
    checked = values_checked = 0
    worst_location = None
    for record in records:

        def trace(name, values):
            nonlocal worst, relative, checked, values_checked, worst_location
            expected = record['traces'][name]
            assert len(values) == len(expected)
            for a, b in zip(values, expected):
                error = abs(a - b)
                if error > worst:
                    worst = error
                    worst_location = dict(name=name, actual=a, reference=b)
                if abs(b) > 0.0001:
                    relative = max(relative, error / abs(b))
                assert math.isfinite(a) and error <= 0.002 + 0.002 * abs(b), (name, a, b, error)
            checked += 1
            values_checked += len(values)
        logits = model.step(record['token'], trace)
        expected = record['logits']
        assert max(range(len(logits)), key=logits.__getitem__) == max(range(len(expected)), key=expected.__getitem__)
        for a, b in zip(logits, expected):
            error = abs(a - b)
            if error > worst:
                worst = error
                worst_location = dict(name='logits', actual=a, reference=b)
            if abs(b) > 0.0001:
                relative = max(relative, error / abs(b))
            assert math.isfinite(a) and error <= 0.002 + 0.002 * abs(b), (a, b, error)
        checked += 1
        values_checked += len(logits)
    print('TRAINED CPU DOUBLE REFERENCE PASS', json.dumps(dict(tokens=len(records), tensor_comparisons=checked, scalar_comparisons=values_checked, max_absolute_error=worst, max_relative_error_abs_reference_gt_1e_4=relative, tolerance='abs <= 0.002 + 0.002*abs(reference)', matching_greedy_token_ids=True, worst_absolute_location=worst_location)), flush=True)
    model.reset()
if __name__ == '__main__':
    import argparse, resource
    from directamd_llm.gpu import GPU
    from directamd_llm.model import Qwen2
    from directamd_llm.tokenizer import Tokenizer
    parser = argparse.ArgumentParser(description='Independent double-precision CPU/GPU check of two trained Qwen tokens')
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--reference-library', required=True, type=Path)
    parser.add_argument('--backend', required=True, type=Path)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    root = args.model.resolve()
    reference_library = args.reference_library.resolve()
    config = Config.load(root / 'config.json')
    tokens = Tokenizer(root).chat("Türkiye'nin başkenti neresidir?")[:2]
    ref = StreamingReference(config)
    references = []
    for token in tokens:
        logits, traces = ref.step(token)
        references.append(dict(token=token, logits=logits, traces=traces))
    gpu = GPU(str(args.backend.resolve()), args.device, 7400 * 1024 ** 2)
    weights = SafeWeights(root).load(gpu, config, 160)
    model = Qwen2(gpu, config, weights, 160)
    compare(model, references)
    print('GPU statistics', json.dumps(gpu.stats(), sort_keys=True))
    print('CPU/GPU validation process max RSS KiB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

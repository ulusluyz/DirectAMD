"""Persistent FP32 tensors backed by DirectAMD AMDGPU buffer objects."""
import array
import ctypes as ct
import math


class Tensor:
    def __init__(self, gpu, handle, count):
        if not handle:
            raise RuntimeError(gpu.error())
        self.gpu = gpu
        self.handle = handle
        self.count = count

    def __del__(self):
        if getattr(self, "handle", None):
            self.gpu.lib.damd_free(self.handle)
            self.handle = None

    def view(self, offset, count):
        return Tensor(self.gpu, self.gpu.lib.damd_view(self.handle, offset, count), count)

    def upload(self, values):
        data = values if isinstance(values, array.array) and values.typecode == "f" else array.array("f", values)
        if len(data) != self.count:
            raise ValueError("Upload shape mismatch")
        pointer = (ct.c_float * len(data)).from_buffer(data)
        self.gpu.check(self.gpu.lib.damd_upload(self.handle, pointer, len(data)))

    def upload_fp16(self, raw):
        if len(raw) != self.count * 2:
            raise ValueError("FP16 upload shape mismatch")
        self.gpu.check(self.gpu.lib.damd_upload_f16(self.handle, raw, self.count))

    def read(self):
        data = array.array("f", [0.0]) * self.count
        pointer = (ct.c_float * len(data)).from_buffer(data)
        self.gpu.check(self.gpu.lib.damd_read(self.handle, pointer, len(data)))
        return data


class GPU:
    def __init__(self, library, device, budget=4 * 1024**3):
        self.lib = ct.CDLL(library)
        p = ct.c_void_p
        u = ct.c_uint32
        z = ct.c_uint64
        f = ct.c_float
        self.lib.damd_error.restype = ct.c_char_p
        signatures = {
            "damd_open": (p, [ct.c_char_p, z]),
            "damd_close": (None, [p]),
            "damd_alloc": (p, [p, z, ct.c_int]),
            "damd_view": (p, [p, z, z]),
            "damd_free": (None, [p]),
            "damd_upload": (ct.c_int, [p, ct.POINTER(f), z]),
            "damd_upload_f16": (ct.c_int, [p, ct.c_void_p, z]),
            "damd_read": (ct.c_int, [p, ct.POINTER(f), z]),
            "damd_binary": (ct.c_int, [ct.c_int, p, p, p, u, u, u]),
            "damd_unary": (ct.c_int, [ct.c_int, p, p, p, f]),
            "damd_rope": (ct.c_int, [p, p, p]),
            "damd_stats": (ct.c_int, [p, ct.POINTER(z)]),
        }
        for name, (result, args) in signatures.items():
            function = getattr(self.lib, name)
            function.restype = result
            function.argtypes = args
        self.handle = self.lib.damd_open(device.encode(), budget)
        if not self.handle:
            raise RuntimeError(self.error())
        self.zeros = {}

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.damd_close(self.handle)
            self.handle = None

    def error(self):
        return self.lib.damd_error().decode("utf-8", "replace")

    def check(self, status):
        if status:
            raise RuntimeError(self.error())

    def tensor(self, count, values=None, vram=False):
        tensor = Tensor(self, self.lib.damd_alloc(self.handle, count, int(vram)), count)
        if values is not None:
            tensor.upload(values)
        return tensor

    def zero(self, count):
        if count not in self.zeros:
            self.zeros[count] = self.tensor(count)
        return self.zeros[count]

    def binary(self, kind, a, b, count, rows=0, columns=0, inner=0):
        output = self.tensor(count)
        self.check(self.lib.damd_binary(kind, a.handle, b.handle, output.handle, rows, columns, inner))
        return output

    def add(self, a, b):
        return self.binary(0, a, b, a.count)

    def multiply(self, a, b):
        return self.binary(1, a, b, a.count)

    def copy(self, source, destination=None):
        if destination is None:
            return self.add(source, self.zero(source.count))
        self.check(self.lib.damd_binary(0, source.handle, self.zero(source.count).handle,
                                       destination.handle, 0, 0, 0))
        return destination

    def gemv(self, matrix, vector, rows):
        return self.binary(2, matrix, vector, rows, rows, vector.count)

    def gemm(self, a, b, rows, columns, inner):
        return self.binary(3, a, b, rows * columns, rows, columns, inner)

    def unary(self, kind, x, aux=None, epsilon=0):
        output = self.tensor(x.count)
        self.check(self.lib.damd_unary(kind, x.handle, output.handle,
                                      aux.handle if aux else None, epsilon))
        return output

    def norm(self, x, weights, epsilon):
        return self.unary(4, x, weights, epsilon)

    def silu(self, x):
        return self.unary(6, x)

    def softmax(self, x):
        return self.unary(7, x)

    def rope(self, x, phase):
        output = self.tensor(x.count)
        self.check(self.lib.damd_rope(x.handle, phase.handle, output.handle))
        return output

    def stats(self):
        data = (ct.c_uint64 * 7)()
        self.check(self.lib.damd_stats(self.handle, data))
        return dict(zip(("resident_bytes", "peak_bytes", "submissions", "fences", "budget_bytes",
                         "vram_tensor_bytes", "gtt_tensor_bytes"), data))

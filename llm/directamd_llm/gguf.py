"""Checked little-endian GGUF v2/v3 metadata and tensor-directory reader."""
import struct, json, math, collections
from pathlib import Path

class GGUF:

    def __init__(self, path):
        self.path = Path(path)
        self.f = self.path.open('rb')
        self.size = self.path.stat().st_size
        if self.f.read(4) != b'GGUF':
            raise ValueError('Not GGUF')
        self.version = self.num('I')
        if self.version not in (2, 3):
            raise ValueError('Unsupported GGUF version')
        nt, nm = (self.num('Q'), self.num('Q'))
        self.meta = {}
        if nt > 100000 or nm > 100000:
            raise ValueError('GGUF table too large')
        for _ in range(nm):
            key = self.string()
            if key in self.meta:
                raise ValueError('Duplicate metadata key')
            self.meta[key] = self.value(self.num('I'))
        self.tensors = []
        for _ in range(nt):
            name = self.string()
            nd = self.num('I')
            if not 1 <= nd <= 4:
                raise ValueError('Unsupported tensor rank')
            dims = [self.num('Q') for _ in range(nd)]
            typ = self.num('I')
            off = self.num('Q')
            self.tensors.append(dict(name=name, dims=dims, type=typ, offset=off))
        align = self.meta.get('general.alignment', 32)
        if type(align) is not int or not 1 <= align <= 4096 or align & align - 1:
            raise ValueError('Invalid GGUF alignment')
        self.start = (self.f.tell() + align - 1) // align * align

    def num(self, fmt):
        n = struct.calcsize('<' + fmt)
        if self.f.tell() + n > min(self.size, 64 * 1024 ** 2):
            raise ValueError('Truncated or oversized GGUF metadata')
        raw = self.f.read(n)
        if len(raw) != n:
            raise ValueError('Truncated GGUF')
        return struct.unpack('<' + fmt, raw)[0]

    def string(self):
        n = self.num('Q')
        if n > 16 * 1024 ** 2 or self.f.tell() + n > min(self.size, 64 * 1024 ** 2):
            raise ValueError('GGUF string too large')
        raw = self.f.read(n)
        if len(raw) != n:
            raise ValueError('Truncated string')
        return raw.decode('utf8')

    def value(self, t):
        formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if t in formats:
            return self.num(formats[t])
        if t == 8:
            return self.string()
        if t == 9:
            typ, n = (self.num('I'), self.num('Q'))
            if n > 2000000:
                raise ValueError('Array too large')
            return [self.value(typ) for _ in range(n)]
        raise ValueError(f'Unknown metadata type {t}')

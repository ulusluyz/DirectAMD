"""Qwen2 byte-level BPE using the local tokenizer.json and the PCRE2 Unicode regex."""
import ctypes as ct
import ctypes.util
import json
import re
from pathlib import Path


class UnicodeSplit:
    def __init__(self, pattern):
        library = ctypes.util.find_library("pcre2-8")
        if not library:
            raise RuntimeError("Qwen tokenizer requires the local libpcre2-8 runtime")
        self.lib = ct.CDLL(library)
        self.lib.pcre2_compile_8.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_uint32,
                                           ct.POINTER(ct.c_int), ct.POINTER(ct.c_size_t), ct.c_void_p]
        self.lib.pcre2_compile_8.restype = ct.c_void_p
        self.lib.pcre2_match_data_create_from_pattern_8.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.pcre2_match_data_create_from_pattern_8.restype = ct.c_void_p
        self.lib.pcre2_match_8.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t, ct.c_size_t,
                                         ct.c_uint32, ct.c_void_p, ct.c_void_p]
        self.lib.pcre2_get_ovector_pointer_8.argtypes = [ct.c_void_p]
        self.lib.pcre2_get_ovector_pointer_8.restype = ct.POINTER(ct.c_size_t)
        self.lib.pcre2_match_data_free_8.argtypes = [ct.c_void_p]
        self.lib.pcre2_code_free_8.argtypes = [ct.c_void_p]
        error, offset = ct.c_int(), ct.c_size_t()
        raw = pattern.encode()
        self.code = self.lib.pcre2_compile_8(raw, len(raw), 0x00080000 | 0x00020000,
                                           ct.byref(error), ct.byref(offset), None)
        if not self.code:
            raise ValueError(f"Tokenizer regex failed at {offset.value}: PCRE2 {error.value}")
        self.match = self.lib.pcre2_match_data_create_from_pattern_8(self.code, None)
        if not self.match:
            raise MemoryError("Cannot allocate tokenizer regex state")

    def __del__(self):
        if getattr(self, "match", None):
            self.lib.pcre2_match_data_free_8(self.match)
        if getattr(self, "code", None):
            self.lib.pcre2_code_free_8(self.code)

    def pieces(self, text):
        raw = text.encode("utf-8")
        start = 0
        while start < len(raw):
            status = self.lib.pcre2_match_8(self.code, raw, len(raw), start, 0, self.match, None)
            if status < 1:
                raise ValueError(f"Tokenizer regex failed to cover the input at byte {start}")
            pair = self.lib.pcre2_get_ovector_pointer_8(self.match)
            begin, end = pair[0], pair[1]
            if begin != start or end <= start:
                raise ValueError("Tokenizer regex left uncovered text")
            yield raw[begin:end]
            start = end


class Tokenizer:
    def __init__(self, directory):
        directory = Path(directory)
        raw = json.loads((directory / "tokenizer.json").read_text())
        settings = json.loads((directory / "tokenizer_config.json").read_text())
        model = raw["model"]
        if model["type"] != "BPE" or model.get("dropout") or model.get("ignore_merges") or raw.get("normalizer"):
            raise ValueError("Unsupported Qwen tokenizer model or normalizer")
        pre = raw["pre_tokenizer"]
        components = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else [pre]
        split = [item for item in components if item.get("type") == "Split"]
        byte = [item for item in components if item.get("type") == "ByteLevel"]
        if (len(split) != 1 or len(byte) != 1 or len(components) != 2 or
                split[0].get("invert") or byte[0].get("add_prefix_space") or
                byte[0].get("use_regex") or raw.get("decoder", {}).get("type") != "ByteLevel"):
            raise ValueError("Unsupported tokenizer pre-tokenizer/decoder")
        self.split = UnicodeSplit(split[0]["pattern"]["Regex"])
        self.vocab = model["vocab"]
        self.inverse = {int(value): key for key, value in self.vocab.items()}
        if len(self.inverse) != len(self.vocab):
            raise ValueError("Duplicate vocabulary IDs")
        self.special = {}
        for item in raw.get("added_tokens", []):
            if item.get("lstrip") or item.get("rstrip") or item.get("single_word"):
                raise ValueError("Unsupported added-token matching rule")
            self.special[item["content"]] = item["id"]
            self.inverse[item["id"]] = item["content"]
        for token in ("<|im_start|>", "<|im_end|>"):
            if token not in self.special:
                raise ValueError("Missing Qwen chat token")
        template = settings.get("chat_template", "")
        if not isinstance(template, str) or not all(value in template for value in (
                "<|im_start|>", "<|im_end|>", "add_generation_prompt",
                "You are Qwen, created by Alibaba Cloud.")):
            raise ValueError("Unsupported chat template; expected standard Qwen2.5 chat")
        self.eos = self.special[settings.get("eos_token", "<|im_end|>")]
        merges = model["merges"]
        self.ranks = {tuple(pair if isinstance(pair, list) else pair.split(" ")): rank
                      for rank, pair in enumerate(merges)}
        # Construct the GPT-2 reversible byte alphabet directly.
        keep = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
        self.byte_encoder = {}
        extra = 0
        for value in range(256):
            if value in keep:
                self.byte_encoder[value] = chr(value)
            else:
                self.byte_encoder[value] = chr(256 + extra)
                extra += 1
        self.byte_decoder = {char: value for value, char in self.byte_encoder.items()}
        self.cache = {}
        self.special_pattern = re.compile("(" + "|".join(re.escape(t) for t in
                                         sorted(self.special, key=len, reverse=True)) + ")")

    def bpe(self, piece):
        text = "".join(self.byte_encoder[value] for value in piece)
        if text in self.cache:
            return self.cache[text]
        parts = list(text)
        while len(parts) > 1:
            candidates = [(self.ranks[pair], pair) for pair in zip(parts, parts[1:]) if pair in self.ranks]
            if not candidates:
                break
            _, chosen = min(candidates)
            merged = []
            index = 0
            while index < len(parts):
                if index + 1 < len(parts) and (parts[index], parts[index + 1]) == chosen:
                    merged.append(parts[index] + parts[index + 1])
                    index += 2
                else:
                    merged.append(parts[index])
                    index += 1
            parts = merged
        try:
            tokens = [self.vocab[part] for part in parts]
        except KeyError as error:
            raise ValueError(f"Tokenizer vocabulary missing byte piece {error}") from error
        self.cache[text] = tokens
        return tokens

    def encode(self, text):
        result = []
        for part in self.special_pattern.split(text):
            if part in self.special:
                result.append(self.special[part])
            elif part:
                for piece in self.split.pieces(part):
                    result.extend(self.bpe(piece))
        return result

    def chat(self, prompt):
        text = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
                "You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" +
                prompt + "<|im_end|>\n<|im_start|>assistant\n")
        return self.encode(text)

    def decode(self, ids):
        output = bytearray()
        for token in ids:
            if token not in self.inverse:
                raise ValueError(f"Token ID {token} missing from tokenizer")
            text = self.inverse[token]
            if text in self.special:
                continue
            try:
                output.extend(self.byte_decoder[char] for char in text)
            except KeyError as error:
                raise ValueError("Tokenizer contains a non-byte token") from error
        return output.decode("utf-8", "replace")

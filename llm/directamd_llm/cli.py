import argparse
import json
import sys
from pathlib import Path

from .gpu import GPU
from .model import Config, Qwen2
from .tokenizer import Tokenizer
from .weights import SafeWeights


def main(library):
    parser = argparse.ArgumentParser(description="Local Qwen2 FP32 inference using DirectAMD DRM")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="/dev/dri/renderD128")
    parser.add_argument("--prompt")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--memory-mib", type=int, default=0, help="GPU tensor budget in MiB; 0 uses VRAM capacity minus 256 MiB")
    parser.add_argument("--self-test", action="store_true", help="untrained synthetic GPU/CPU integration check")
    args = parser.parse_args()
    try:
        if args.self_test:
            from .verify import run
            run(library, args.device)
            return 0
        if args.model is None or args.prompt is None:
            parser.error("--model and --prompt are required")
        if args.max_tokens < 1 or args.memory_mib < 0:
            raise ValueError("Token count must be positive and memory budget nonnegative")
        config = Config.load(args.model / "config.json")
        if not 1 <= args.context_length <= config.max_position_embeddings:
            raise ValueError("Context length outside model limits")
        tokenizer = Tokenizer(args.model)
        prompt_ids = tokenizer.chat(args.prompt)
        if len(prompt_ids) + args.max_tokens > args.context_length:
            raise ValueError("Prompt and generation exceed --context-length")
        gpu = GPU(library, args.device, (args.memory_mib * 1024**2) if args.memory_mib else (1 << 63))
        weights = SafeWeights(args.model).load(gpu, config, args.context_length)
        model = Qwen2(gpu, config, weights, args.context_length)
        eos = {tokenizer.eos}
        generation_file = args.model / "generation_config.json"
        if generation_file.is_file():
            generation = json.loads(generation_file.read_text())
            ids = generation.get("eos_token_id", [])
            eos.update(ids if isinstance(ids, list) else [ids])
        generated, timing = model.generate(prompt_ids, args.max_tokens, eos)
        print(tokenizer.decode(generated))
        print(json.dumps({"device": args.device, "prompt_tokens": len(prompt_ids),
                          **timing, **gpu.stats()}, sort_keys=True), file=sys.stderr)
        return 0
    except (RuntimeError, ValueError, KeyError, OSError, AssertionError) as error:
        print(f"DirectAMD LLM error: {str(error) or type(error).__name__}", file=sys.stderr)
        return 1

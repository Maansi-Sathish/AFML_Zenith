# 6 - Memory-efficient LMSYS data preparation

import argparse
import hashlib
import json
import os
import random
import time
import tempfile

import numpy as np

SPECIALS = ["<|pad|>", "<|eos|>", "<|user|>", "<|assistant|>"]


def get_hf_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def conversation_to_text(conv):
    parts = []
    for m in conv:
        content = (m.get("content") or "").strip()
        if not content:
            return None
        tag = "<|user|>" if m["role"] == "user" else "<|assistant|>"
        parts.append(tag + content)
    return "".join(parts) + "<|eos|>"


def train_tokenizer(text_file, vocab_size):
    from tokenizers import Tokenizer, models, trainers
    from tokenizers import pre_tokenizers, decoders

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def text_iterator():
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    tok.train_from_iterator(text_iterator(), trainer)
    return tok


def tokenize_file(tok, text_file, output_file, max_tokens, chunk=256):
    total = 0
    batch = []

    with open(output_file, "wb") as out:
        def write_batch(texts):
            nonlocal total
            if not texts or total >= max_tokens:
                return

            encodings = tok.encode_batch(texts)

            for enc in encodings:
                if total >= max_tokens:
                    break

                ids = enc.ids[:max_tokens - total]
                arr = np.asarray(ids, dtype=np.uint16)
                out.write(arr.tobytes())
                total += len(arr)

        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                batch.append(json.loads(line))

                if len(batch) >= chunk:
                    write_batch(batch)
                    batch = []

                if total >= max_tokens:
                    break

        if batch and total < max_tokens:
            write_batch(batch)

    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/kaggle/working/data")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--train_tokens", type=int, default=60_000_000)
    ap.add_argument("--eval_tokens", type=int, default=1_000_000)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--keep_redacted", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    start = time.time()

    from datasets import load_dataset

    print("Loading LMSYS dataset in streaming mode...")

    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        streaming=True,
        token=get_hf_token()
    )

    ds = ds.select_columns([
        "conversation", "language",
        "openai_moderation", "redacted"
    ])

    if args.sample:
        ds = ds.take(args.sample)
        print(f"Processing first {args.sample:,} rows...")
    else:
        print("Processing full dataset...")

    stats = dict(
        total=0, not_english=0, flagged=0,
        redacted=0, bad_length=0,
        duplicate=0, kept=0
    )

    seen = set()

    with tempfile.TemporaryDirectory(
        dir=args.out_dir
    ) as temp_dir:

        paths = {
            name: os.path.join(temp_dir, name + ".jsonl")
            for name in ["train", "val", "test"]
        }

        files = {
            name: open(path, "w", encoding="utf-8")
            for name, path in paths.items()
        }

        try:
            for ex in ds:
                stats["total"] += 1

                if ex["language"] != "English":
                    stats["not_english"] += 1
                    continue

                mods = ex.get("openai_moderation") or []
                if any(
                    isinstance(m, dict) and m.get("flagged", False)
                    for m in mods
                ):
                    stats["flagged"] += 1
                    continue

                if ex.get("redacted") and not args.keep_redacted:
                    stats["redacted"] += 1
                    continue

                conv = ex["conversation"]
                text = conversation_to_text(conv)

                if text is None or not (20 <= len(text) <= 8000):
                    stats["bad_length"] += 1
                    continue

                key = hashlib.md5(
                    conv[0]["content"].strip().lower().encode("utf-8")
                ).hexdigest()

                if key in seen:
                    stats["duplicate"] += 1
                    continue

                seen.add(key)

                bucket = int(key, 16) % 100
                split = (
                    "val" if bucket == 0
                    else "test" if bucket == 1
                    else "train"
                )

                files[split].write(json.dumps(text) + "\n")
                stats["kept"] += 1

                if stats["total"] % 50000 == 0:
                    print(
                        f"Processed {stats['total']:,} rows | "
                        f"Kept {stats['kept']:,}"
                    )
        finally:
            for f in files.values():
                f.close()

        print("\nFiltering complete:", stats)

        print("\nTraining shared tokenizer...")
        tok = train_tokenizer(paths["train"], args.vocab)
        tok.save(os.path.join(args.out_dir, "tokenizer.json"))

        print("Tokenizer vocabulary:", tok.get_vocab_size())

        counts = {}

        for name, budget in [
            ("train", args.train_tokens),
            ("val", args.eval_tokens),
            ("test", args.eval_tokens)
        ]:
            print(f"\nTokenizing {name}...")

            counts[name] = tokenize_file(
                tok,
                paths[name],
                os.path.join(args.out_dir, name + ".bin"),
                budget
            )

            print(f"{name}: {counts[name]:,} tokens")

    meta = dict(
        vocab_size=tok.get_vocab_size(),
        token_counts=counts,
        filter_stats=stats,
        seed=args.seed,
        specials=SPECIALS,
        dtype="uint16",
        requested=dict(
            train=args.train_tokens,
            eval=args.eval_tokens
        )
    )

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"\nCompleted in {(time.time()-start)/60:.1f} minutes."
    )
    print("Output directory:", args.out_dir)


if __name__ == "__main__":
    main()

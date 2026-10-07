"""The ``tweetmood`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings, load_settings
from .models.base import CLASSIC, MODEL_NAMES

DEFAULT_MODELS = list(CLASSIC)


def _split_dir(args, settings: Settings) -> Path:
    return Path(args.split or settings.data_dir / "prepared")


def _model_dir(args, settings: Settings) -> Path:
    return Path(args.models or settings.model_dir)


def _device(settings: Settings) -> str:
    if settings.device != "auto":
        return settings.device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _model_options(names, settings: Settings, embedder: str) -> dict:
    opts = {}
    for name in names:
        if name.startswith("hybrid-"):
            spec = ({"kind": "hash", "dim": 64, "max_len": 64} if embedder == "hash" else
                    {"kind": "hf", "model_name": settings.hf_model, "max_len": 64, "offline": settings.hf_offline})
            opts[name] = {"embedder": spec, "device": _device(settings)}
        if name == "finetune":
            opts[name] = {"model_name": settings.hf_model, "local_files_only": settings.hf_offline,
                          "device": _device(settings)}
    return opts


def cmd_synth(args, settings) -> int:
    from .data.synthetic import make_tweets

    df = make_tweets(n=args.n, seed=settings.seed if args.seed is None else args.seed)
    out = Path(args.out or settings.data_dir / "raw" / "synthetic_tweets.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} tweets to {out}")
    return 0


def cmd_prepare(args, settings) -> int:
    from .data.loaders import load_many
    from .data.split import deduplicate, make_split, save_split

    raw, reports = load_many(args.source, sample=args.sample, seed=settings.seed)
    for r in reports:
        print(f"{r.source}: read {r.rows_read}, kept {r.rows_kept}, dropped {r.dropped}")
    _, dedup = deduplicate(raw)
    df = make_split(raw, seed=settings.seed, holdout_sources=tuple(args.holdout_source or ()))
    manifest = save_split(df, _split_dir(args, settings),
                          {"seed": settings.seed, "dedup": dedup, "holdout_sources": args.holdout_source or []})
    print(json.dumps({k: manifest[k] for k in ("fingerprint", "rows", "dedup")}, indent=2))
    return 0


def cmd_train(args, settings) -> int:
    from .data.split import load_split
    from .workflow import train_models

    df, manifest = load_split(_split_dir(args, settings))
    names = args.model or DEFAULT_MODELS
    train_models(df, manifest, names, _model_dir(args, settings), seed=settings.seed,
                 model_options=_model_options(names, settings, args.embedder))
    return 0


def cmd_evaluate(args, settings) -> int:
    from .data.split import load_split
    from .workflow import evaluate_models, to_markdown

    df, manifest = load_split(_split_dir(args, settings))
    report = evaluate_models(df, manifest, _model_dir(args, settings), n_boot=args.n_boot)
    print(to_markdown(report))
    return 0


def cmd_predict(args, settings) -> int:
    from .workflow import predict

    texts = list(args.text) or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    for row in predict(texts, _model_dir(args, settings), model_name=args.model):
        print(json.dumps(row, ensure_ascii=False))
    return 0


def cmd_normalize(args, settings) -> int:
    from .text.normalize import Normalizer

    norm = Normalizer(negation_scope=args.negation_scope)
    for t in args.text:
        print(norm(t))
    return 0


def cmd_ablate(args, settings) -> int:
    from .data.split import load_split
    from .workflow import ablate

    df, _ = load_split(_split_dir(args, settings))
    table = ablate(df, model=args.model, seed=settings.seed)
    print(table.to_string(index=False))
    return 0


def cmd_demo(args, settings) -> int:
    from .data.split import deduplicate, make_split, save_split
    from .data.synthetic import make_tweets
    from .workflow import ablate, evaluate_models, predict, to_markdown, train_models

    out = Path(args.out or settings.model_dir / "demo")
    raw = make_tweets(n=args.n, seed=settings.seed)
    _, dedup = deduplicate(raw)
    df = make_split(raw, seed=settings.seed, holdout_sources=("synthetic_genz",))
    manifest = save_split(df, out / "prepared", {"seed": settings.seed, "dedup": dedup})
    print(f"synthetic tweets: {len(raw)} -> {manifest['rows']} (synthetic_genz is test-only), dedup {dedup}")
    names = DEFAULT_MODELS + (["hybrid-bilstm", "hybrid-transformer"] if args.neural else [])
    train_models(df, manifest, names, out / "models", seed=settings.seed,
                 model_options=_model_options(names, settings, "hash"))
    print(to_markdown(evaluate_models(df, manifest, out / "models", n_boot=args.n_boot)))
    print("normalizer ablation (tfidf-logreg):")
    print(ablate(df, seed=settings.seed).to_string(index=False))
    for row in predict(["this new album is mid ngl \U0001F644", "that pizza was bussin fr \U0001F525",
                        "I don't love it :(", "not bad at all :)"], out / "models"):
        print(json.dumps(row, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tweetmood", description="Negation-safe, slang-aware tweet sentiment.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write synthetic tweets (classic and Gen-Z sources)")
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seed", type=int)
    p.add_argument("--out")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("prepare", help="load, de-duplicate and write the one shared split")
    p.add_argument("--source", action="append", required=True, metavar="KIND=PATH",
                   help="sentiment140 | tweeteval | social | unified")
    p.add_argument("--sample", type=int, help="class-balanced Sentiment140 sample, without replacement")
    p.add_argument("--holdout-source", action="append", help="put this source in the test split only")
    p.add_argument("--split")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="train models on the shared split and select the best on validation")
    p.add_argument("--model", action="append", choices=MODEL_NAMES)
    p.add_argument("--embedder", choices=["hash", "hf"], default="hash", help="embedder for the hybrid models")
    p.add_argument("--split")
    p.add_argument("--models")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score every trained model on the test split once")
    p.add_argument("--split")
    p.add_argument("--models")
    p.add_argument("--n-boot", type=int, default=1000)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="label tweets with the selected model (arguments or stdin)")
    p.add_argument("text", nargs="*")
    p.add_argument("--model", choices=MODEL_NAMES, help="default: the best model on validation")
    p.add_argument("--models")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("normalize", help="show the normalized text")
    p.add_argument("text", nargs="+")
    p.add_argument("--negation-scope", action="store_true")
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("ablate", help="train one TF-IDF model per normalizer variant")
    p.add_argument("--model", default="tfidf-logreg", choices=["tfidf-logreg", "tfidf-svm", "tfidf-nb"])
    p.add_argument("--split")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("demo", help="offline end-to-end demo on synthetic tweets")
    p.add_argument("--out")
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--neural", action="store_true", help="add the hybrid models with the hash embedder (torch)")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args, load_settings())


if __name__ == "__main__":
    raise SystemExit(main())

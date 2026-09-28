from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform

from . import __version__

def doctor() -> int:
    available = True
    checks = {
        "decisionbench_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import llama_cpp

        checks["llama_cpp"] = llama_cpp.__version__
    except ImportError:
        checks["llama_cpp"] = None
        available = False
    print(json.dumps(checks, indent=2))
    return 0 if available else 1


def run(args) -> int:
    from .engine import CpuEngine
    from .runner import run_benchmark

    engine = CpuEngine(
        args.gguf,
        model_id=args.model_id,
        threads=args.threads,
        context_tokens=args.context_tokens,
        chat_format=args.chat_format,
    )
    run_benchmark(args.input, args.output, engine)
    return 0


def report(args) -> int:
    from .report import create_comparison_report, create_report

    if args.runs:
        create_comparison_report(
            args.runs,
            args.output,
            title=args.title,
            warning=args.warning,
        )
    else:
        create_report(args.gold, args.predictions, args.output)
    return 0


def fetch(args) -> int:
    if args.fetch_kind == "datasets":
        from .datasets import build_core

        result = build_core(args.cache, args.owned_base, args.output)
        print(json.dumps(result, indent=2))
    else:
        from .models import fetch_model

        print(fetch_model(args.manifest, args.cache))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="decisionbench")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check the local CPU benchmark environment")
    fetcher = subparsers.add_parser("fetch", help="Fetch pinned datasets or a model")
    fetchers = fetcher.add_subparsers(dest="fetch_kind", required=True)
    dataset_fetcher = fetchers.add_parser("datasets", help="Build the frozen core benchmark")
    dataset_fetcher.add_argument("--cache", type=Path, required=True)
    dataset_fetcher.add_argument("--owned-base", type=Path, required=True)
    dataset_fetcher.add_argument("--output", type=Path, required=True)
    model_fetcher = fetchers.add_parser("model", help="Download and verify one pinned GGUF")
    model_fetcher.add_argument("--manifest", type=Path, required=True)
    model_fetcher.add_argument("--cache", type=Path, required=True)
    runner = subparsers.add_parser("run", help="Score one JSONL benchmark with one local GGUF")
    runner.add_argument("--model-id", required=True)
    runner.add_argument("--gguf", type=Path, required=True)
    runner.add_argument("--input", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--threads", type=int)
    runner.add_argument("--context-tokens", type=int, default=4096)
    runner.add_argument("--chat-format")
    reporter = subparsers.add_parser("report", help="Build static reports from row-level predictions")
    mode = reporter.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runs", type=Path)
    mode.add_argument("--predictions", type=Path)
    reporter.add_argument("--gold", type=Path)
    reporter.add_argument("--title", default="DecisionBench comparison")
    reporter.add_argument(
        "--warning",
        default="Preliminary results; do not treat a small suite as the full benchmark.",
    )
    reporter.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        raise SystemExit(doctor())
    if args.command == "fetch":
        if args.fetch_kind == "datasets" and args.output.exists():
            parser.error("Output must be new")
        raise SystemExit(fetch(args))
    if args.command == "report":
        if args.predictions and not args.gold:
            parser.error("--gold is required with --predictions")
        if args.runs and args.gold:
            parser.error("--gold cannot be used with --runs")
        raise SystemExit(report(args))
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LongQA-only evaluation/generation entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

logger = logging.getLogger(__name__)

DEFAULT_CFG = "egolongqa"
X9_LONGQA_ROOT = "/Volumes/Crucial X9/theory-of-mind"
DEFAULT_PREDS = f"output/{DEFAULT_CFG}/predictions.jsonl"
DEFAULT_OUTPUT = f"output/{DEFAULT_CFG}/results.json"


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def _first_existing(*paths: str) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


DEFAULT_GOLDEN = _first_existing(
    os.path.join(X9_LONGQA_ROOT, "wearable_ai_2026_egolongqa_val_700.jsonl"),
    _resolve_path("../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"),
)
DEFAULT_VIDEO_FOLDER = _first_existing(
    os.path.join(X9_LONGQA_ROOT, "egolongqa_merged_val"),
    os.path.join(X9_LONGQA_ROOT, "egolongqa/val"),
    _resolve_path("../egolongqa/val"),
)


def load_jsonl(path: str) -> list[dict[str, object]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_results(output_path: str, results: dict[str, object]) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = {k: v for k, v in results.items() if k != "per_row"}
    base, ext = os.path.splitext(output_path)
    summary_path = f"{base}_summary{ext or '.json'}"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def normalize_answer(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""

    upper = raw.upper()
    if upper in ("A", "B", "C", "D"):
        return upper

    m = re.match(r"^([A-Da-d])\s*[\.:\)]", raw)
    if m:
        return m.group(1).upper()

    m = re.match(r"^\(([A-Da-d])\)", raw)
    if m:
        return m.group(1).upper()

    m = re.match(r"^(?:option|answer)\s*[:.]?\s*([A-Da-d])\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b(?:is|answer)\s*[:.]?\s*([A-Da-d])\s*\.?\s*$", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-Da-d])\b", raw)
    if m:
        return m.group(1).upper()

    return upper[0] if upper[0] in "ABCD" else ""


def evaluate_longqa(
    golden: list[dict[str, object]],
    preds: list[dict[str, object]],
) -> dict[str, object]:
    if len(golden) != len(preds):
        raise ValueError(
            f"Golden ({len(golden)}) and predictions ({len(preds)}) "
            f"must have the same number of entries."
        )

    results_per_row: list[dict[str, object]] = []
    correct = 0

    for i, (g, p) in enumerate(zip(golden, preds)):
        gold = str(g["mcq_answer"]).strip().upper()
        pred_raw = str(p.get("mcq_answer", ""))
        pred = normalize_answer(pred_raw)
        is_correct = pred == gold
        if is_correct:
            correct += 1

        results_per_row.append(
            {
                "index": i,
                "video_path": g.get("video_path", ""),
                "question": g.get("question", ""),
                "gold_answer": gold,
                "pred_raw": pred_raw,
                "pred_parsed": pred,
                "correct": is_correct,
                "category": g.get("category", ""),
            }
        )

    total = len(preds)
    accuracy = correct / total if total else 0.0
    category_stats: dict[str, dict[str, int]] = {}
    for row in results_per_row:
        cat = str(row["category"])
        category_stats.setdefault(cat, {"correct": 0, "total": 0})
        category_stats[cat]["total"] += 1
        if row["correct"]:
            category_stats[cat]["correct"] += 1

    category_accuracy = {
        cat: round(v["correct"] / v["total"], 4) for cat, v in sorted(category_stats.items())
    }

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "category_accuracy": category_accuracy,
        "per_row": results_per_row,
    }


def _run_longqa(
    golden_path: str,
    preds_path: str,
    output_path: str,
) -> None:
    golden = load_jsonl(golden_path)
    preds = load_jsonl(preds_path)
    if len(golden) != len(preds):
        logger.warning(
            "Golden and predictions differ in length (golden=%d, predictions=%d)",
            len(golden),
            len(preds),
        )
        raise ValueError("Golden and predictions must have the same number of entries")

    results = evaluate_longqa(golden, preds)
    summary_path = write_results(output_path, results)

    print(f"LongQA Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    if results.get("category_accuracy"):
        print("  Per-category accuracy:")
        for cat, acc in results["category_accuracy"].items():
            label = cat if cat else "(no category)"
            print(f"    {label}: {acc:.4f}")
    print(f"Results written to {output_path}")
    print(f"Summary written to {summary_path}")


def _run_generation(args: argparse.Namespace) -> None:
    from run_generate_longqa import main as generate_main

    argv = [
        "run_generate_longqa.py",
        "--input",
        _resolve_path(args.golden),
        "--output",
        _resolve_path(args.predictions),
        "--video-folder",
        _resolve_path(args.video_folder),
        "--model-type",
        args.model_type,
        "--backend",
        args.backend,
        "--max-frames",
        str(args.max_frames),
    ]

    if args.llm_model:
        argv.extend(["--llm-model", args.llm_model])
    if args.batch_size is not None:
        argv.extend(["--batch-size", str(args.batch_size)])
    if args.max_samples is not None:
        argv.extend(["--max-samples", str(args.max_samples)])
    if args.num_gpus is not None:
        argv.extend(["--num-gpus", str(args.num_gpus)])
    if args.tp is not None:
        argv.extend(["--tp", str(args.tp)])
    if args.concurrency != 16:
        argv.extend(["--concurrency", str(args.concurrency)])
    if args.no_eval:
        argv.append("--no-eval")

    old = sys.argv
    sys.argv = argv
    try:
        generate_main()
    finally:
        sys.argv = old


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongQA-only runner")
    parser.add_argument(
        "--task",
        type=str,
        choices=["longqa"],
        default="longqa",
        help="Only longqa is supported in this trimmed branch.",
    )
    parser.add_argument(
        "--golden",
        type=str,
        default=DEFAULT_GOLDEN,
        help="Path to Golden JSONL.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=DEFAULT_PREDS,
        help="Path to LongQA predictions JSONL.",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default=DEFAULT_VIDEO_FOLDER,
        help="Folder containing LongQA video files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to evaluation output JSON.",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Alias for --output.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate existing predictions only (no generation).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Generate predictions only (no eval).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="qwen",
        choices=["llama4", "qwen", "openai", "gemini"],
        help="Model type for generation.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override model id (OpenAI users: set to gpt-4o-mini/gpt-4o).",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm", "openai", "gemini"],
        default="hf",
        help=(
            "Inference backend: 'hf', 'vllm', 'openai', or 'gemini' "
            "(default: hf)."
        ),
    )
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    return parser

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = _build_parser()
    args = parser.parse_args()

    if args.eval_only and args.no_eval:
        parser.error("--eval-only and --no-eval are mutually exclusive")

    golden = _resolve_path(args.golden)
    predictions = _resolve_path(args.predictions)
    output = _resolve_path(args.output or args.eval_output or DEFAULT_OUTPUT)

    if args.video_folder:
        args.video_folder = _resolve_path(args.video_folder)

    if args.eval_only:
        _run_longqa(golden, predictions, output)
        return

    _run_generation(args)
    if not args.no_eval:
        _run_longqa(golden, predictions, output)


if __name__ == "__main__":
    main()

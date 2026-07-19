# Wearable AI - LongQA (EgoLongQA) Starter Kit

This folder contains a LongQA-only evaluation flow for `egolongqa`.

## What’s included

- `run_evaluation.py` — LongQA-only entry point (generation + evaluation)
- `run_generate_longqa.py` — LongQA prediction generator
- `model.py` — model + data-loader utilities
- `slurm_runner.py` — optional multi-node SLURM generation helper
- `requirements.txt`, `requirements-dev.txt`

## Defaults

- Golden annotations: `/Volumes/Crucial X9/theory-of-mind/wearable_ai_2026_egolongqa_val_700.jsonl` (auto-falls back to `../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl` if the volume is missing)
- Video folder: `/Volumes/Crucial X9/theory-of-mind/egolongqa_merged_val` (fallback: `../egolongqa/val`)
- Predictions output: `output/egolongqa/predictions.jsonl`
- Evaluation output: `output/egolongqa/results.json`

## Quick start

From inside `wearable-ai-lite2/starter_kit`:

```bash
python run_evaluation.py --task longqa --backend vllm
```

This generates predictions (if not already present), evaluates them, and writes:
- `output/egolongqa/predictions.jsonl`
- `output/egolongqa/results.json`
- `output/egolongqa/results_summary.json`

## OpenAI API backend

Use this for routing directly to OpenAI image-capable models (default `gpt-4o-mini`).

1. Add your key to `wearable-ai-lite2/starter_kit/.env`:
   - `OPENAI_API_KEY=<your key>`
2. Run:

```bash
python run_evaluation.py --task longqa --backend openai --model-type openai --llm-model gpt-4o-mini
```

Optional:
- `OPENAI_API_BASE` to point to a custom endpoint
- `OPENAI_ORGANIZATION` for org-scoped credentials
- `OPENAI_TIMEOUT_SECONDS` to change request timeout

## Gemini API backend

Use this for routing to Gemini image models (for example `gemini-2.5-flash-lite`).

1. Add your key to `wearable-ai-lite2/starter_kit/.env`:
   - `GEMINI_API_KEY=<your key>`
   - Optional: `GEMINI_API_BASE=https://generativelanguage.googleapis.com`
   - Optional: `GEMINI_API_VERSION=v1beta`
   - Optional: `GEMINI_TIMEOUT_SECONDS=1200`
2. Run:

```bash
python run_evaluation.py --task longqa --backend gemini --model-type gemini --llm-model gemini-2.5-flash-lite
```

## Eval only

If predictions already exist:

```bash
python run_evaluation.py --task longqa --eval-only
```

## Generate only

```bash
python run_evaluation.py --task longqa --no-eval
```

## Alternate invocation

Use LongQA generator directly:

```bash
python run_generate_longqa.py --video-folder /Volumes/Crucial\\ X9/theory-of-mind/egolongqa_merged_val --model-type qwen
```
```bash
python run_generate_longqa.py --backend openai --model-type openai --llm-model gpt-4o-mini --video-folder /Volumes/Crucial\\ X9/theory-of-mind/egolongqa_merged_val
```

## Important

- This variant intentionally strips ConvQA/Proactive logic.
- All non-`longqa` paths and flags from the original combined starter kit are removed.

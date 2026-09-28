# DecisionBench

DecisionBench measures how well open models up to 4B parameters work as typed
decision engines on an ordinary CPU.

The model receives evidence, a question, and named options. DecisionBench maps
the options to single-token labels, reads the model's next-token logits, and
returns a probability distribution without generating an answer.

This repository is intentionally small. Version 1 contains a CPU benchmark CLI,
fixed benchmark manifests, row-level results, and static reports. It does not
train models, host weights, or reproduce Jev's private architecture.

## What it measures

The frozen core v1 suite has 1,071 rows:

- 231 public JevBench rows;
- 120 BANKING77 rows across 12 intents;
- 200 balanced BoolQ rows;
- 150 balanced WANLI rows;
- 120 MASSIVE rows across 12 intents and five languages;
- 70 MMLU-Pro rows across 14 domains;
- 36 project-authored base cases;
- four robustness variants for each authored case, giving 180 authored rows.

The six planned model families are Qwen3.5 4B, Gemma 3 4B, Phi-4 Mini,
Llama 3.2 3B, MiniCPM5, and Granite 3.3. Exact GGUF revisions and hashes are in
`models/`.

The benchmark method and six-model JevBench pilot are complete. The full
1,071-row matrix is still running.

## Metrics

- **Coverage**: valid predictions / all declared rows. Unit: fraction from 0 to
  1. Higher is
  better. Invalid or missing predictions stay in the denominator.
- **Accuracy**: correct top choices / all declared rows. Unit: fraction from 0
  to 1. Higher is better.
- **Macro-F1**: F1 is calculated for each answer class, then averaged. Unit:
  score from 0 to 1. It gives small classes equal weight.
- **NLL**: mean negative log probability assigned to the correct option. Unit:
  natural-log loss per row. Lower is better.
- **Brier score**: mean squared error of the full option-probability vector.
  Unit: squared probability error per row. Lower is better.
- **ECE**: confidence gap across ten bins, weighted by rows in each bin. Unit:
  probability difference from 0 to 1. Lower is better.
- **Argmax flip rate**: changed top choices / valid paired robustness cases.
  Unit: fraction from 0 to 1. Lower is better.
- **Probability movement**: mean absolute probability change after semantic
  option alignment. Unit: probability difference from 0 to 1. Lower is better.
- **Missing-evidence error rate**: wrong or invalid predictions / authored
  missing-evidence rows. Unit: fraction from 0 to 1. Lower is better.
- **Latency**: median and 95th-percentile wall time per decision. Unit: seconds.
  Lower is faster.
- **Throughput**: completed timed decisions / total decision time. Unit:
  decisions per second. Higher is faster.
- **Peak process RAM**: highest sampled memory used by the benchmark process
  during a case. Unit: bytes.
- **Allowed-token mass**: full-vocabulary probability assigned to the declared
  aliases. Unit: probability from 0 to 1. Higher means the model more strongly
  obeyed the answer format.

All reported probabilities are conditional on the declared option aliases.
They are not automatically real-world confidence.

## Smoke result

The seven-row smoke suite checks installation, prompt rendering, alias
compatibility, and end-to-end evidence output. It is too small for headline
quality claims.

| Model | Coverage | Accuracy | Allowed mass | p50 seconds | Peak RAM GiB |
|---|---:|---:|---:|---:|---:|
| Gemma 3 4B | 1.000 | 0.714 | 1.000 | 8.226 | 5.39 |
| Granite 3.3 2B | 1.000 | 0.571 | 0.758 | 5.262 | 3.15 |
| Llama 3.2 3B | 1.000 | 0.714 | 1.000 | 7.979 | 4.24 |
| MiniCPM5 | 1.000 | 0.857 | 0.977 | 5.230 | 2.98 |
| Phi-4 Mini | 1.000 | 0.857 | 0.916 | 9.572 | 4.92 |
| Qwen3.5 4B | 1.000 | 0.857 | 0.999 | 10.617 | 5.25 |

Row-level evidence and checksums are under `results/smoke-v1/`.

## JevBench pilot result

Pilot v1 contains 30 fixed public JevBench rows: 10 easy, 10 original, and 10
hard. This is an early comparison, not the full benchmark.

| Model | Coverage | Accuracy | Macro-F1 | Allowed mass |
|---|---:|---:|---:|---:|
| Gemma 3 4B | 1.000 | 0.633 | 0.603 | 1.000 |
| Granite 3.3 2B | 0.967 | 0.667 | 0.662 | 0.931 |
| Llama 3.2 3B | 1.000 | 0.600 | 0.627 | 1.000 |
| MiniCPM5 | 1.000 | 0.667 | 0.710 | 0.903 |
| Phi-4 Mini | 1.000 | 0.633 | 0.651 | 0.973 |
| Qwen3.5 4B | 1.000 | 0.667 | 0.713 | 0.998 |

Granite exceeded the fixed 4,096-token budget on one hard row, so that row
remains an explicit failure. Timing and RAM from this pilot are diagnostic only
because the full CPU benchmark was running at the same time.

Row-level predictions, summaries, and checksums are under
`results/jevbench-pilot-v1/`.

## Commands

```powershell
decisionbench doctor
decisionbench fetch datasets --cache dataset-cache\sources `
  --owned-base benchmarks\owned\base36.jsonl `
  --output benchmarks\core-v1.jsonl
decisionbench fetch model --manifest models\qwen3.5-4b.json `
  --cache models-cache
decisionbench run --model-id MODEL_ID --gguf MODEL.gguf `
  --input benchmarks\smoke-v1.jsonl --output runs\MODEL_ID
decisionbench report --gold benchmarks\smoke-v1.jsonl `
  --predictions runs\MODEL_ID\predictions.jsonl --output reports\MODEL_ID
```

`run` writes `manifest.json`, `predictions.jsonl`, `summary.json`, and
`SHA256SUMS`. If an incomplete run is started again with the same benchmark and
model hashes, only missing row IDs are scored.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m pip install llama-cpp-python==0.3.35 `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\decisionbench doctor
```

The extra index provides the prebuilt Windows CPU wheel. Model weights remain
upstream and are never committed to this repository.

See `METHOD.md` for the frozen prompt and scoring method, `THIRD_PARTY.md` for
source licenses, and `CONTRIBUTING.md` for the small extension points.

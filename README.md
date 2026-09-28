# DecisionBench

DecisionBench measures how well small open models work as typed decision engines
on CPU. Most models are marketed at 2-4B parameters; effective-size models are
identified separately, not treated as having the same total size or RAM needs.

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

The active lineup is Qwen3.5 4B, Gemma 3 4B, **Gemma 4 E2B**, Phi-4 Mini,
Llama 3.2 3B, and MiniCPM5. E2B has no results yet. Its official QAT Q4_0 artifact
is different from the other models' Q4_K_M builds. E2B describes effective size,
not its total checkpoint parameter count. Exact GGUF revisions and hashes are
in `models/`. Granite is retired; its manifest and original results remain
available as historical evidence. E4B is not in scope.

Published results are preliminary. The full 1,071-row matrix is still running
and is not included in the published evidence catalogue.

## Metrics

- **Coverage**: valid predictions / all declared rows. Unit: fraction from 0 to
  1. Higher is
  better. Invalid or missing predictions stay in the denominator.
- **Accuracy**: correct top choices / all declared rows. Unit: fraction from 0
  to 1. Higher is better.
- **Macro-F1**: F1 is calculated for each declared answer class, then averaged.
  Unit: score from 0 to 1, higher is better. Only BANKING77, BoolQ, WANLI, and
  MASSIVE have supported fixed class sets; no mixed-suite or JevBench F1.
- **NLL**: mean negative log probability assigned to the correct option. Unit:
  natural-log loss per row. Lower is better.
- **Brier score**: mean squared error of the full option-probability vector.
  Unit: squared probability error per row. Lower is better.
- **ECE**: confidence gap across ten bins, weighted by valid rows in each bin.
  Unit: probability difference from 0 to 1. Lower is better. Always read it
  with coverage; NLL and Brier are omitted if any declared prediction is invalid.
- **Argmax flip rate**: changed top choices / valid paired robustness cases.
  Unit: fraction from 0 to 1. Lower is better.
- **Valid pair coverage**: valid original/variant pairs / declared pairs.
  Unit: fraction from 0 to 1. Higher is better; invalid pairs are not stability.
- **Probability movement**: mean absolute probability change after semantic
  option alignment, averaged over options in valid pairs. Unit: probability
  difference from 0 to 1. Lower is better.
- **Missing-evidence error rate**: wrong or invalid predictions / authored
  missing-evidence rows. Unit: fraction from 0 to 1. Lower is better.
- **Latency**: median and 95th-percentile wall time per decision. Unit: seconds.
  Lower is faster.
- **Throughput**: completed timed decisions / total decision time. Unit:
  decisions per second. Higher is faster.
- **Peak process RAM**: highest process memory sample during model evaluation.
  Unit: bytes; not model size or an end-to-end/load peak.
- **Allowed-token mass**: full-vocabulary probability assigned to the declared
  aliases. Unit: probability from 0 to 1. Higher means the model more strongly
  obeyed the answer format.

All reported probabilities are conditional on the declared option aliases.
They are not automatically real-world confidence.

## Smoke status

The seven-row smoke suite checks prompt rendering, alias compatibility, and
evidence output. The five previously evaluated active models passed all seven
rows without inference errors; E2B is pending. This is an installation check,
not a quality or speed ranking. Results are in `results/smoke-v1/`; the current
report is `results/smoke-v1-summary-r2/REPORT.md`.

## JevBench pilot result

Pilot v1 contains 30 fixed public JevBench rows: 10 easy, 10 original, and 10
hard. These are first-in-file selections, not random or representative samples.
The original tier has only five paired groups. Counts below mean **correct
answers / declared rows in that tier**. All five evaluated active models have
30/30 valid predictions; this small pilot does not establish a winner.

| Model | Easy | Original | Hard |
|---|---:|---:|---:|
| Gemma 3 4B | 10/10 | 7/10 | 2/10 |
| Gemma 4 E2B | Not run | Not run | Not run |
| Llama 3.2 3B | 10/10 | 4/10 | 4/10 |
| MiniCPM5 | 10/10 | 6/10 | 4/10 |
| Phi-4 Mini | 10/10 | 6/10 | 3/10 |
| Qwen3.5 4B | 10/10 | 6/10 | 4/10 |

Timing and RAM are diagnostic only: the pilot shared the CPU with another run.
Original evidence is under `results/jevbench-pilot-v1/`; the current report is
`results/jevbench-pilot-v1-summary-r2/REPORT.md`.

## Evidence correction

The September 28, 2026 audit found that Git had changed line endings in
checksummed files. The correction restores the original hashed bytes and tells
Git to preserve frozen evidence exactly. **No prediction values or gold labels
were changed or rescored.** Earlier reports remain historical; use the `-r2`
reports, which exclude Granite and remove mixed-task rankings.

The frozen BANKING77 subset also had 110 incorrect source-row references.
`benchmarks/corrections/core-v1-banking77.json` supplies the correct references
without changing the benchmark used by the ongoing run. New conversions fix
the references and therefore must use a new output path.

The authored cases are 36 bases, not 180 independent examples. Human review is
unconfirmed; the frozen `author-reviewed` tag is not proof of human review.
Some cases rely on unstated routing assumptions. Treat them as exploratory,
not a separately validated quality benchmark.

Check the complete published catalogue, file hashes, model identity, prediction
consistency, and recomputed metrics without downloading models:

```powershell
.\.venv\Scripts\decisionbench verify
```

Historical manifests lack full machine/runtime details and exact upstream
conversion provenance. The evaluated GGUF hashes are pinned; unknown source
revisions are not retroactively invented. See `METHOD.md` for remaining limits.

## Commands

```powershell
decisionbench doctor
decisionbench fetch datasets --cache dataset-cache\sources `
  --owned-base benchmarks\owned\base36.jsonl `
  --output benchmarks\core-v2-local.jsonl
decisionbench fetch model --manifest models\qwen3.5-4b.json `
  --cache models-cache
decisionbench run --model-id MODEL_ID --gguf MODEL.gguf `
  --input benchmarks\smoke-v1.jsonl --output runs\MODEL_ID
decisionbench report --gold benchmarks\smoke-v1.jsonl `
  --predictions runs\MODEL_ID\predictions.jsonl --output reports\MODEL_ID
```

`run` writes `manifest.json`, `predictions.jsonl`, `summary.json`, and
`SHA256SUMS`. New incomplete runs can resume only with matching model, prompt,
implementation, settings, and runtime records. Legacy or truncated runs fail
explicitly instead of mixing evidence. Outputs and report directories are
create-only.

## Smallest useful demo

A local terminal example is the next step: give evidence, a question, and named
options as JSON; show the selected ID and conditional probabilities, then show
what happens when the options are reversed. It needs no website or server.
This is a proposed demo, not an existing command: inference must first be
separated from the benchmark schema so users do not need a gold answer.

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

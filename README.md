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
Llama 3.2 3B, and MiniCPM5. E2B's official QAT Q4_0 artifact
is different from the other models' Q4_K_M builds. E2B describes effective size,
not its total checkpoint parameter count. Exact GGUF revisions and hashes are
in `models/`. Granite is retired; its manifest and original results remain
available as historical evidence. E4B is not in scope.

The full core v1 matrix completed on September 28, 2026: all six active models
produced valid predictions for all 1,071 rows, totaling 6,426 decisions. Complete
row-level evidence and the combined report are included in the publication
catalogue. This remains an exploratory quality study, not a controlled speed
comparison or a direct evaluation against Jev.

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

## Full core v1 results

Each cell is **correct answers / all declared rows for that task**. A larger
fraction is better. Coverage is 1.000 for every active model on every task:
there were no runtime-error or invalid-prediction rows, not zero wrong answers.

| Task | Gemma 3 4B | Gemma 4 E2B | Llama 3.2 3B | MiniCPM5 2B | Phi-4 Mini | Qwen3.5 4B |
|---|---:|---:|---:|---:|---:|---:|
| JevBench easy | 48/48 | 48/48 | 44/48 | 47/48 | 48/48 | 48/48 |
| JevBench original | 57/72 | 56/72 | 36/72 | 50/72 | 50/72 | 62/72 |
| JevBench hard | 40/111 | 49/111 | 43/111 | 46/111 | 56/111 | 61/111 |
| BANKING77 subset | 92/120 | 92/120 | 92/120 | 90/120 | 97/120 | 98/120 |
| BoolQ subset | 150/200 | 166/200 | 140/200 | 170/200 | 166/200 | 177/200 |
| WANLI subset | 59/150 | 65/150 | 50/150 | 73/150 | 86/150 | 101/150 |
| MASSIVE subset | 94/120 | 101/120 | 88/120 | 91/120 | 94/120 | 106/120 |
| MMLU-Pro subset | 22/70 | 18/70 | 12/70 | 25/70 | 22/70 | 37/70 |
| Authored robustness (exploratory) | 152/180 | 155/180 | 133/180 | 138/180 | 158/180 | 170/180 |

The combined report, including per-task probability metrics and robustness,
is `results/core-v1-active-six-summary/REPORT.md` with JSON and CSV beside it.
Raw runs are under `results/core-v1/` and `results/core-v1-e2b/`.
Granite's already-completed run is kept as historical evidence only, outside
the active table and the 6,426-decision count.

The subsets are fixed first-in-file selections, not full-dataset scores.
Authored variants are related and lack confirmed human review. E2B uses a
different quantization, and older runs lack the newer runtime/implementation
records. No combined accuracy ranking or controlled speed/RAM claim is made.

## Smoke status

The seven-row smoke suite checks prompt rendering, alias compatibility, and
evidence output. All six active models passed all seven rows without inference
errors. This is an installation check, not a quality or speed ranking.
Results are in `results/smoke-v1/` and `results/e2b-smoke-v1/`; the current
combined report is `results/smoke-v1-active-six-summary/REPORT.md`.

## Earlier JevBench pilot

Pilot v1 contains 30 fixed public JevBench rows: 10 easy, 10 original, and 10
hard. These are first-in-file selections, not random or representative samples.
The original tier has only five paired groups. Counts below mean **correct
answers / declared rows in that tier**. All five evaluated active models have
30/30 valid predictions; this small pilot does not establish a winner.
E2B was not run on this separate pilot; its later full-suite results are above.

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
were changed or rescored.** Earlier reports remain historical. Use the
`core-v1-active-six-summary` and `smoke-v1-active-six-summary` reports for the
completed active lineup, and the `jevbench-pilot-v1-summary-r2` report for the
earlier pilot. These exclude Granite and remove mixed-task rankings.

The frozen BANKING77 subset also had 110 incorrect source-row references.
`benchmarks/corrections/core-v1-banking77.json` supplies the correct references
without changing the benchmark used by the completed runs. New conversions fix
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

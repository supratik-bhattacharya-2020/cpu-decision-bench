# Method

## Decision rule

Each row contains evidence, one question, and 2-26 stable option IDs. The prompt
maps the ordered options to `A`, `B`, `C`, and so on.

The frozen instruction is:

```text
Apply the question to the evidence.
Choose exactly one listed option.
Respond with only its option label.
```

The GGUF's embedded chat template renders the messages. Templates that insert
the current date receive the fixed benchmark date `2026-09-24`. Thinking mode
is disabled when the template supports that flag.

The engine checks that each alias is exactly one token at the answer boundary.
It performs one forward pass, reads the final-position vocabulary logits,
selects the alias logits, and applies softmax only across those values.

## Evidence

Every prediction stores the rendered prompt and hash, model and GGUF hash,
option IDs, alias token IDs, raw option logits, conditional probabilities,
full-vocabulary top token, allowed-token mass, token count, timing, and sampled
peak process RAM.

Errors are written as rows and remain in metric denominators. Outputs are
create-only except that a new-format incomplete run may append missing row IDs
after its benchmark, model, prompt implementation, settings, and runtime match.
New manifests record CPU/OS, Python/package versions, llama.cpp build details,
and a normalized-source fingerprint for the inference implementation. Legacy
runs do not contain all of this information and cannot be safely resumed by
the new runner. A truncated final prediction or leftover writer lock requires
manual recovery into a new run; it is never silently discarded.

Reports verify all required checksums, complete row coverage (including explicit
error rows), model identity, probability/argmax/logit consistency, and prompt
text hashes before recomputing metrics. New summaries must exactly match the
recomputation. Historical accuracy, coverage, NLL, Brier, and ECE are checked;
historical mixed-class F1 and robustness summaries are superseded.

## Core suite v1

`benchmarks/core-v1.jsonl` contains:

- all 231 public JevBench v1.4.0 rows at revision
  `2fa63fa3226cb369795525ed011800f57dcbd894`;
- 120 BANKING77 test rows: the first 10 rows for each of 12 frozen intents;
- 200 BoolQ validation rows: the first 100 rows per answer;
- 150 WANLI test rows: the first 50 rows per mapped relation;
- 120 MASSIVE v1.1 test rows: the first two rows for each of 12 frozen
  intents in English, Spanish, German, Hindi, and Chinese;
- 70 MMLU-Pro test rows: the first five rows in each of 14 domains;
- 36 project-authored base cases;
- option-reversal, question-rewording, irrelevant-context, and
  missing-evidence variants for every authored case.

The authored labels were set before model runs. The frozen `author-reviewed`
tag has no named human reviewer or supporting review record. Human adjudication
is unconfirmed. There are only 36 independent bases; variants are related rows.
Some routing rules are implicit, and some missing-evidence forms remove a
policy definition as well as evidence. Do not edit labels after seeing model
results; a reviewed replacement must be a separate, versioned suite.

BANKING77 core v1 records a selected-row ordinal in `source_row` for 110 rows
instead of the original CSV row index. The supplement
`benchmarks/corrections/core-v1-banking77.json` maps every selected ID to its
original zero-based CSV data-row index. It pins the raw-source hash and checks
that evidence, questions, option order and gold labels are unchanged. New
conversions use the correct index; the frozen v1 benchmark is not rewritten.
To independently check the supplement against an already downloaded source:

```powershell
decisionbench verify --banking77-source dataset-cache\core-v1\public\banking77-test.csv
```

BANKING77 and MASSIVE each have more than 26 intent classes. Core v1 uses 12
frozen intents from each because the alias contract supports at most 26 options.
This limit is explicit; the results are not full-dataset intent scores.

## JevBench pilot v1

The time-bounded pilot takes the first 10 frozen rows from each public
JevBench tier: easy, original, and hard. It contains 30 rows, with 180 historical
model decisions including the retired Granite run. These are not representative
tier samples: easy contains 10 intent rows, original contains 10 policy rows
in five paired groups, and hard contains eight long-policy and two probability
rows. Small differences do not justify model rankings.

The pilot uses a fixed 4,096-token context budget. A prompt that exceeds this
budget is recorded as an invalid prediction and remains in the denominator.
The pilot was run while the full CPU matrix continued, so its latency and RAM
numbers are diagnostic rather than controlled performance comparisons.

## Timing

Model load time is reported separately. Case latency includes prompt rendering,
tokenization, alias checks, the forward pass, and result creation. It excludes
model loading and dataset download.

RAM is process RSS sampled every 10 ms during evaluation, not total model
memory or a complete load-time peak. `logits_all=True` also retains intermediate
token logits; this is not a final-logit-only memory optimization.
The pilot ran concurrently with another CPU job. An uncommitted full-run Llama
row records about 57.5 hours for one decision; the cause was not established.
Do not clip or delete it to improve a speed result. All existing timing and
memory measurements are diagnostic; controlled performance runs are still needed.

## Metrics and revisions

Accuracy and coverage include all declared rows. NLL uses natural logarithms
with a probability floor of 1e-12. Multiclass Brier sums squared error across
options, then averages over rows. NLL and Brier are omitted when any prediction
is invalid. ECE uses ten equal-width probability bins over valid predictions.
Macro-F1 averages over all declared classes (zero for an undefined class F1),
only in BANKING77, BoolQ, WANLI and MASSIVE. Answer positions in MMLU-Pro and
context-dependent labels in JevBench are not common semantic classes.

Robustness aligns stable option IDs, counts flips over valid pairs, and reports
valid-pair coverage over all declared meaning-preserving variants. Probability
movement averages absolute changes over options in those pairs. Wrong, invalid
and absent missing-evidence predictions count as errors.

Existing GGUF revisions and hashes identify the exact evaluated files, but
upstream checkpoint revisions used by third-party quantizers were not recorded.
Do not use today's upstream revision as a claim about an old conversion.
Gemma 4 E2B has a pinned official QAT Q4_0 artifact and no evaluation results
yet. Its effective-size label is distinct from total checkpoint parameters.

The September 28 correction preserves historical evidence bytes (including
CRLF) with `.gitattributes`; new outputs explicitly use UTF-8/LF. Old report
directories are retained for traceability, not endorsed as current analysis.
The `-r2` reports are recomputed from the same predictions, with no inference.
`results/publication.json` declares the complete published evidence set.

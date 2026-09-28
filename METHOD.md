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
create-only except that an incomplete run may append missing row IDs after its
benchmark and model hashes match.

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

The authored labels were set before model runs. `author-reviewed` means the case
was checked for internal consistency by its author; it does not mean
independent human adjudication.

BANKING77 and MASSIVE each have more than 26 intent classes. Core v1 uses 12
frozen intents from each because the alias contract supports at most 26 options.
This limit is explicit; the results are not full-dataset intent scores.

## JevBench pilot v1

The time-bounded pilot takes the first 10 frozen rows from each public
JevBench tier: easy, original, and hard. It contains 30 rows and 180 total
model decisions.

The pilot uses a fixed 4,096-token context budget. A prompt that exceeds this
budget is recorded as an invalid prediction and remains in the denominator.
The pilot was run while the full CPU matrix continued, so its latency and RAM
numbers are diagnostic rather than controlled performance comparisons.

## Timing

Model load time is reported separately. Case latency includes prompt rendering,
tokenization, alias checks, the forward pass, and result creation. It excludes
model loading and dataset download.

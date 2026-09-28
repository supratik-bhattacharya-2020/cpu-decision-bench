# DecisionBench JevBench pilot - corrected active lineup

First 10 rows per tier, not a representative sample or a model ranking. Timing and RAM were measured alongside another CPU run. Granite is archived; E2B has not run.

Accuracy = correct answers / all declared rows; coverage = valid predictions / all rows.
Both are fractions from 0 to 1; higher is better. Failures remain in the denominator.
F1 is shown only for tasks sharing one label set, not across unrelated questions.
Timing and RAM remain diagnostic data in raw runs, not performance rankings.

| Model | Dataset | Correct / rows | Coverage | Accuracy |
|---|---|---:|---:|---:|
| gemma-3-4b-it-q4_k_m | jevbench-easy | 10 / 10 | 1.000 | 1.000 |
| gemma-3-4b-it-q4_k_m | jevbench-hard | 2 / 10 | 1.000 | 0.200 |
| gemma-3-4b-it-q4_k_m | jevbench-original | 7 / 10 | 1.000 | 0.700 |
| gemma-4-e2b-it-qat-q4_0 | Not run | n/a | n/a | n/a |
| llama-3.2-3b-instruct-q4_k_m | jevbench-easy | 10 / 10 | 1.000 | 1.000 |
| llama-3.2-3b-instruct-q4_k_m | jevbench-hard | 4 / 10 | 1.000 | 0.400 |
| llama-3.2-3b-instruct-q4_k_m | jevbench-original | 4 / 10 | 1.000 | 0.400 |
| minicpm5-2b-q4_k_m | jevbench-easy | 10 / 10 | 1.000 | 1.000 |
| minicpm5-2b-q4_k_m | jevbench-hard | 4 / 10 | 1.000 | 0.400 |
| minicpm5-2b-q4_k_m | jevbench-original | 6 / 10 | 1.000 | 0.600 |
| phi-4-mini-instruct-q4_k_m | jevbench-easy | 10 / 10 | 1.000 | 1.000 |
| phi-4-mini-instruct-q4_k_m | jevbench-hard | 3 / 10 | 1.000 | 0.300 |
| phi-4-mini-instruct-q4_k_m | jevbench-original | 6 / 10 | 1.000 | 0.600 |
| qwen3.5-4b-q4_k_m | jevbench-easy | 10 / 10 | 1.000 | 1.000 |
| qwen3.5-4b-q4_k_m | jevbench-hard | 4 / 10 | 1.000 | 0.400 |
| qwen3.5-4b-q4_k_m | jevbench-original | 6 / 10 | 1.000 | 0.600 |

# DecisionBench smoke compatibility - corrected active lineup

Seven compatibility rows, not a quality benchmark. Historical timing and RAM are diagnostic only. Granite is archived; E2B has not run.

Accuracy = correct answers / all declared rows; coverage = valid predictions / all rows.
Both are fractions from 0 to 1; higher is better. Failures remain in the denominator.
F1 is shown only for tasks sharing one label set, not across unrelated questions.
Timing and RAM remain diagnostic data in raw runs, not performance rankings.

| Model | Dataset | Correct / rows | Coverage | Accuracy |
|---|---|---:|---:|---:|
| gemma-3-4b-it-q4_k_m | banking77-12-intent | 0 / 1 | 1.000 | 0.000 |
| gemma-3-4b-it-q4_k_m | boolq-balanced | 0 / 1 | 1.000 | 0.000 |
| gemma-3-4b-it-q4_k_m | jevbench-easy | 1 / 1 | 1.000 | 1.000 |
| gemma-3-4b-it-q4_k_m | massive-5-language-12-intent | 1 / 1 | 1.000 | 1.000 |
| gemma-3-4b-it-q4_k_m | mmlu-pro-domain-balanced | 1 / 1 | 1.000 | 1.000 |
| gemma-3-4b-it-q4_k_m | owned-robustness | 1 / 1 | 1.000 | 1.000 |
| gemma-3-4b-it-q4_k_m | wanli-balanced | 1 / 1 | 1.000 | 1.000 |
| gemma-4-e2b-it-qat-q4_0 | Not run | n/a | n/a | n/a |
| llama-3.2-3b-instruct-q4_k_m | banking77-12-intent | 1 / 1 | 1.000 | 1.000 |
| llama-3.2-3b-instruct-q4_k_m | boolq-balanced | 0 / 1 | 1.000 | 0.000 |
| llama-3.2-3b-instruct-q4_k_m | jevbench-easy | 1 / 1 | 1.000 | 1.000 |
| llama-3.2-3b-instruct-q4_k_m | massive-5-language-12-intent | 1 / 1 | 1.000 | 1.000 |
| llama-3.2-3b-instruct-q4_k_m | mmlu-pro-domain-balanced | 0 / 1 | 1.000 | 0.000 |
| llama-3.2-3b-instruct-q4_k_m | owned-robustness | 1 / 1 | 1.000 | 1.000 |
| llama-3.2-3b-instruct-q4_k_m | wanli-balanced | 1 / 1 | 1.000 | 1.000 |
| minicpm5-2b-q4_k_m | banking77-12-intent | 0 / 1 | 1.000 | 0.000 |
| minicpm5-2b-q4_k_m | boolq-balanced | 1 / 1 | 1.000 | 1.000 |
| minicpm5-2b-q4_k_m | jevbench-easy | 1 / 1 | 1.000 | 1.000 |
| minicpm5-2b-q4_k_m | massive-5-language-12-intent | 1 / 1 | 1.000 | 1.000 |
| minicpm5-2b-q4_k_m | mmlu-pro-domain-balanced | 1 / 1 | 1.000 | 1.000 |
| minicpm5-2b-q4_k_m | owned-robustness | 1 / 1 | 1.000 | 1.000 |
| minicpm5-2b-q4_k_m | wanli-balanced | 1 / 1 | 1.000 | 1.000 |
| phi-4-mini-instruct-q4_k_m | banking77-12-intent | 0 / 1 | 1.000 | 0.000 |
| phi-4-mini-instruct-q4_k_m | boolq-balanced | 1 / 1 | 1.000 | 1.000 |
| phi-4-mini-instruct-q4_k_m | jevbench-easy | 1 / 1 | 1.000 | 1.000 |
| phi-4-mini-instruct-q4_k_m | massive-5-language-12-intent | 1 / 1 | 1.000 | 1.000 |
| phi-4-mini-instruct-q4_k_m | mmlu-pro-domain-balanced | 1 / 1 | 1.000 | 1.000 |
| phi-4-mini-instruct-q4_k_m | owned-robustness | 1 / 1 | 1.000 | 1.000 |
| phi-4-mini-instruct-q4_k_m | wanli-balanced | 1 / 1 | 1.000 | 1.000 |
| qwen3.5-4b-q4_k_m | banking77-12-intent | 0 / 1 | 1.000 | 0.000 |
| qwen3.5-4b-q4_k_m | boolq-balanced | 1 / 1 | 1.000 | 1.000 |
| qwen3.5-4b-q4_k_m | jevbench-easy | 1 / 1 | 1.000 | 1.000 |
| qwen3.5-4b-q4_k_m | massive-5-language-12-intent | 1 / 1 | 1.000 | 1.000 |
| qwen3.5-4b-q4_k_m | mmlu-pro-domain-balanced | 1 / 1 | 1.000 | 1.000 |
| qwen3.5-4b-q4_k_m | owned-robustness | 1 / 1 | 1.000 | 1.000 |
| qwen3.5-4b-q4_k_m | wanli-balanced | 1 / 1 | 1.000 | 1.000 |

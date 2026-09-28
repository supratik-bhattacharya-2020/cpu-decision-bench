# DecisionBench core v1 - six active models

Per-task exploratory results, not a universal model ranking. Includes Gemma 4 E2B from its separate run directory; excludes retired Granite. Historical and new runs have different implementation/runtime records. Timing and RAM are diagnostic only because runs overlapped or were interrupted. Authored cases lack confirmed human review; BANKING77 source-row corrections are supplied separately.

Accuracy = correct answers / all declared rows; coverage = valid predictions / all rows.
Both are fractions from 0 to 1; higher is better. Failures remain in the denominator.
F1 is shown only for tasks sharing one label set, not across unrelated questions.
Timing and RAM remain diagnostic data in raw runs, not performance rankings.

| Model | Dataset | Correct / rows | Coverage | Accuracy |
|---|---|---:|---:|---:|
| gemma-3-4b-it-q4_k_m | banking77-12-intent | 92 / 120 | 1.000 | 0.767 |
| gemma-3-4b-it-q4_k_m | boolq-balanced | 150 / 200 | 1.000 | 0.750 |
| gemma-3-4b-it-q4_k_m | jevbench-easy | 48 / 48 | 1.000 | 1.000 |
| gemma-3-4b-it-q4_k_m | jevbench-hard | 40 / 111 | 1.000 | 0.360 |
| gemma-3-4b-it-q4_k_m | jevbench-original | 57 / 72 | 1.000 | 0.792 |
| gemma-3-4b-it-q4_k_m | massive-5-language-12-intent | 94 / 120 | 1.000 | 0.783 |
| gemma-3-4b-it-q4_k_m | mmlu-pro-domain-balanced | 22 / 70 | 1.000 | 0.314 |
| gemma-3-4b-it-q4_k_m | owned-robustness | 152 / 180 | 1.000 | 0.844 |
| gemma-3-4b-it-q4_k_m | wanli-balanced | 59 / 150 | 1.000 | 0.393 |
| gemma-4-e2b-it-qat-q4_0 | banking77-12-intent | 92 / 120 | 1.000 | 0.767 |
| gemma-4-e2b-it-qat-q4_0 | boolq-balanced | 166 / 200 | 1.000 | 0.830 |
| gemma-4-e2b-it-qat-q4_0 | jevbench-easy | 48 / 48 | 1.000 | 1.000 |
| gemma-4-e2b-it-qat-q4_0 | jevbench-hard | 49 / 111 | 1.000 | 0.441 |
| gemma-4-e2b-it-qat-q4_0 | jevbench-original | 56 / 72 | 1.000 | 0.778 |
| gemma-4-e2b-it-qat-q4_0 | massive-5-language-12-intent | 101 / 120 | 1.000 | 0.842 |
| gemma-4-e2b-it-qat-q4_0 | mmlu-pro-domain-balanced | 18 / 70 | 1.000 | 0.257 |
| gemma-4-e2b-it-qat-q4_0 | owned-robustness | 155 / 180 | 1.000 | 0.861 |
| gemma-4-e2b-it-qat-q4_0 | wanli-balanced | 65 / 150 | 1.000 | 0.433 |
| llama-3.2-3b-instruct-q4_k_m | banking77-12-intent | 92 / 120 | 1.000 | 0.767 |
| llama-3.2-3b-instruct-q4_k_m | boolq-balanced | 140 / 200 | 1.000 | 0.700 |
| llama-3.2-3b-instruct-q4_k_m | jevbench-easy | 44 / 48 | 1.000 | 0.917 |
| llama-3.2-3b-instruct-q4_k_m | jevbench-hard | 43 / 111 | 1.000 | 0.387 |
| llama-3.2-3b-instruct-q4_k_m | jevbench-original | 36 / 72 | 1.000 | 0.500 |
| llama-3.2-3b-instruct-q4_k_m | massive-5-language-12-intent | 88 / 120 | 1.000 | 0.733 |
| llama-3.2-3b-instruct-q4_k_m | mmlu-pro-domain-balanced | 12 / 70 | 1.000 | 0.171 |
| llama-3.2-3b-instruct-q4_k_m | owned-robustness | 133 / 180 | 1.000 | 0.739 |
| llama-3.2-3b-instruct-q4_k_m | wanli-balanced | 50 / 150 | 1.000 | 0.333 |
| minicpm5-2b-q4_k_m | banking77-12-intent | 90 / 120 | 1.000 | 0.750 |
| minicpm5-2b-q4_k_m | boolq-balanced | 170 / 200 | 1.000 | 0.850 |
| minicpm5-2b-q4_k_m | jevbench-easy | 47 / 48 | 1.000 | 0.979 |
| minicpm5-2b-q4_k_m | jevbench-hard | 46 / 111 | 1.000 | 0.414 |
| minicpm5-2b-q4_k_m | jevbench-original | 50 / 72 | 1.000 | 0.694 |
| minicpm5-2b-q4_k_m | massive-5-language-12-intent | 91 / 120 | 1.000 | 0.758 |
| minicpm5-2b-q4_k_m | mmlu-pro-domain-balanced | 25 / 70 | 1.000 | 0.357 |
| minicpm5-2b-q4_k_m | owned-robustness | 138 / 180 | 1.000 | 0.767 |
| minicpm5-2b-q4_k_m | wanli-balanced | 73 / 150 | 1.000 | 0.487 |
| phi-4-mini-instruct-q4_k_m | banking77-12-intent | 97 / 120 | 1.000 | 0.808 |
| phi-4-mini-instruct-q4_k_m | boolq-balanced | 166 / 200 | 1.000 | 0.830 |
| phi-4-mini-instruct-q4_k_m | jevbench-easy | 48 / 48 | 1.000 | 1.000 |
| phi-4-mini-instruct-q4_k_m | jevbench-hard | 56 / 111 | 1.000 | 0.505 |
| phi-4-mini-instruct-q4_k_m | jevbench-original | 50 / 72 | 1.000 | 0.694 |
| phi-4-mini-instruct-q4_k_m | massive-5-language-12-intent | 94 / 120 | 1.000 | 0.783 |
| phi-4-mini-instruct-q4_k_m | mmlu-pro-domain-balanced | 22 / 70 | 1.000 | 0.314 |
| phi-4-mini-instruct-q4_k_m | owned-robustness | 158 / 180 | 1.000 | 0.878 |
| phi-4-mini-instruct-q4_k_m | wanli-balanced | 86 / 150 | 1.000 | 0.573 |
| qwen3.5-4b-q4_k_m | banking77-12-intent | 98 / 120 | 1.000 | 0.817 |
| qwen3.5-4b-q4_k_m | boolq-balanced | 177 / 200 | 1.000 | 0.885 |
| qwen3.5-4b-q4_k_m | jevbench-easy | 48 / 48 | 1.000 | 1.000 |
| qwen3.5-4b-q4_k_m | jevbench-hard | 61 / 111 | 1.000 | 0.550 |
| qwen3.5-4b-q4_k_m | jevbench-original | 62 / 72 | 1.000 | 0.861 |
| qwen3.5-4b-q4_k_m | massive-5-language-12-intent | 106 / 120 | 1.000 | 0.883 |
| qwen3.5-4b-q4_k_m | mmlu-pro-domain-balanced | 37 / 70 | 1.000 | 0.529 |
| qwen3.5-4b-q4_k_m | owned-robustness | 170 / 180 | 1.000 | 0.944 |
| qwen3.5-4b-q4_k_m | wanli-balanced | 101 / 150 | 1.000 | 0.673 |

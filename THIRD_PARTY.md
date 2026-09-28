# Third-party sources

## JevBench

- Repository: `fstandhartinger/jevbench`
- Revision: `2fa63fa3226cb369795525ed011800f57dcbd894`
- Files: `datasets/public/easy.jsonl`, `original.jsonl`, and `hard.jsonl`
- License: MIT
- Use: all 231 public rows are converted to the common DecisionBench schema.

## Model files

Model weights are not stored here. The exact upstream and GGUF sources are
listed in `models/*.json`.

Gemma 3 uses the Gemma license. The pinned official Gemma 4 E2B model metadata
specifies Apache-2.0. Llama uses the Llama 3.2 Community License.
Users may need to accept upstream terms before downloading those models.
Qwen3.5, MiniCPM5, and Granite manifests identify Apache-2.0 sources. Phi-4 Mini
uses MIT.

Granite's manifest is retained under `models/retired/`; it is not in the active
comparison. Existing third-party GGUF pins identify the evaluated artifact,
not a proven upstream conversion revision. Do not infer that missing revision.

Each user is responsible for following the model and dataset terms that apply
to their use.

## BANKING77

- Source: `PolyAI-LDN/task-specific-datasets`
- Revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- File: `banking_data/test.csv`
- License: CC-BY-4.0

## BoolQ

- Source: `google/boolq`
- Revision: `35b264d03638db9f4ce671b711558bf7ff0f80d5`
- File: `data/validation-00000-of-00001.parquet`
- License: CC-BY-SA-3.0

## WANLI

- Source: `alisawuffles/WANLI`
- Revision: `61c95318fd71c55b6ba355d76253254615f387ec`
- File: `test.jsonl`
- License: CC-BY-4.0

## MASSIVE

- Source metadata: `AmazonScience/massive`
- Revision: `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`
- Archive: `amazon-massive-dataset-1.1.tar.gz`
- Archive SHA-256: `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`
- License: CC-BY-4.0

## MMLU-Pro

- Source: `TIGER-Lab/MMLU-Pro`
- Revision: `b189ec765aa7ed75c8acfea42df31fdae71f97be`
- File: `data/test-00000-of-00001.parquet`
- License: MIT

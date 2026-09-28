from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

import pyarrow.parquet as parquet

from .schema import read_jsonl, sha256_file, sha256_text, validate_row, write_jsonl_create

JEVBENCH_REVISION = "2fa63fa3226cb369795525ed011800f57dcbd894"
JEVBENCH_FILES = {
    "easy": "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    "original": "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    "hard": "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
}
BANKING77_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BANKING77_TEST_SHA256 = "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"
BOOLQ_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
BOOLQ_SHA256 = "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8"
WANLI_REVISION = "61c95318fd71c55b6ba355d76253254615f387ec"
WANLI_SHA256 = "4276e0af7fcdf657d1ab7beb54eaf025fda592a76c9ee86b63b7871953fc74fd"
MASSIVE_REVISION = "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
MASSIVE_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_PRO_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"

BANKING77_INTENTS = [
    "card_arrival",
    "card_delivery_estimate",
    "card_not_working",
    "cash_withdrawal_not_recognised",
    "cash_withdrawal_charge",
    "transaction_charged_twice",
    "card_payment_not_recognised",
    "declined_card_payment",
    "pending_card_payment",
    "request_refund",
    "Refund_not_showing_up",
    "lost_or_stolen_card",
]
MASSIVE_INTENTS = [
    "alarm_set",
    "alarm_remove",
    "weather_query",
    "play_music",
    "calendar_set",
    "calendar_query",
    "email_sendemail",
    "lists_createoradd",
    "transport_taxi",
    "cooking_recipe",
    "qa_definition",
    "news_query",
]
MASSIVE_LOCALES = ["en-US", "es-ES", "de-DE", "hi-IN", "zh-CN"]


def _download(url: str, destination: Path, expected_sha256: str, limit: int = 16 * 1024 * 1024) -> None:
    if destination.exists():
        raise ValueError(f"Refusing to replace {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "decisionbench/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Download exceeded {limit} bytes: {url}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Source changed for {url}: expected {expected_sha256}, received {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def fetch_jevbench(cache: Path) -> dict[str, Path]:
    result = {}
    for split, expected in JEVBENCH_FILES.items():
        destination = cache / "jevbench" / JEVBENCH_REVISION / f"{split}.jsonl"
        if not destination.exists():
            url = (
                "https://raw.githubusercontent.com/fstandhartinger/jevbench/"
                f"{JEVBENCH_REVISION}/datasets/public/{split}.jsonl"
            )
            _download(url, destination, expected)
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Cached JevBench {split} hash changed")
        result[split] = destination
    return result


def _source_path(
    cache: Path,
    name: str,
    url: str,
    sha256: str,
    *,
    limit: int = 16 * 1024 * 1024,
) -> Path:
    destination = cache / "public" / name
    if not destination.exists():
        _download(url, destination, sha256, limit=limit)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"Cached source hash changed: {destination}")
    return destination


def fetch_public_sources(cache: Path) -> dict[str, Path]:
    sources = {}
    sources["banking77"] = _source_path(
        cache,
        "banking77-test.csv",
        (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{BANKING77_COMMIT}/banking_data/test.csv"
        ),
        BANKING77_TEST_SHA256,
    )
    sources["boolq"] = _source_path(
        cache,
        "boolq-validation.parquet",
        (
            "https://huggingface.co/datasets/google/boolq/resolve/"
            f"{BOOLQ_REVISION}/data/validation-00000-of-00001.parquet"
        ),
        BOOLQ_SHA256,
    )
    sources["wanli"] = _source_path(
        cache,
        "wanli-test.jsonl",
        f"https://huggingface.co/datasets/alisawuffles/WANLI/resolve/{WANLI_REVISION}/test.jsonl",
        WANLI_SHA256,
    )
    sources["massive"] = _source_path(
        cache,
        "amazon-massive-dataset-1.1.tar.gz",
        "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
        MASSIVE_SHA256,
        limit=64 * 1024 * 1024,
    )
    sources["mmlu_pro"] = _source_path(
        cache,
        "mmlu-pro-test.parquet",
        (
            "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/"
            f"{MMLU_PRO_REVISION}/data/test-00000-of-00001.parquet"
        ),
        MMLU_PRO_SHA256,
    )
    return sources


def _description(identifier: str) -> str:
    return identifier.replace("_", " ").replace("?", "").strip().capitalize()


def _balanced(rows: list[dict], label_key: str, labels: list[str], per_label: int) -> list[dict]:
    selected = []
    for label in labels:
        matches = [row for row in rows if str(row[label_key]) == label]
        if len(matches) < per_label:
            raise ValueError(f"Need {per_label} rows for {label}; received {len(matches)}")
        selected.extend(matches[:per_label])
    return selected


def convert_banking77(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        source_rows = [
            {**row, "source_row": index}
            for index, row in enumerate(csv.DictReader(stream))
        ]
    selected = _balanced(source_rows, "category", BANKING77_INTENTS, 10)
    options = [{"id": label, "description": _description(label)} for label in BANKING77_INTENTS]
    rows = []
    for source_index, source in enumerate(selected):
        row = {
            "id": f"banking77/test/{source['category']}/{source_index}",
            "dataset": "banking77-12-intent",
            "group_id": source["category"],
            "state": source["text"],
            "question": "Which banking intent best matches the customer message?",
            "options": copy.deepcopy(options),
            "label": source["category"],
            "provenance": {
                "source": "PolyAI-LDN/task-specific-datasets",
                "revision": BANKING77_COMMIT,
                "source_split": "test",
                "source_row": source["source_row"],
                "license": "CC-BY-4.0",
                "selection": "first 10 test rows for each of 12 frozen intents",
            },
        }
        validate_row(row)
        rows.append(row)
    return rows


def banking77_correction(core_path: Path, source_path: Path) -> dict:
    if sha256_file(source_path) != BANKING77_TEST_SHA256:
        raise ValueError("BANKING77 correction requires the pinned source CSV")
    frozen = {
        row["id"]: row for row in read_jsonl(core_path)
        if row["dataset"] == "banking77-12-intent"
    }
    corrected = convert_banking77(source_path)
    if set(frozen) != {row["id"] for row in corrected}:
        raise ValueError("BANKING77 correction row set differs")
    mapping = []
    for row in corrected:
        old = frozen[row["id"]]
        old_copy = copy.deepcopy(old)
        old_copy["provenance"]["source_row"] = row["provenance"]["source_row"]
        if old_copy != row:
            raise ValueError(f"Correction changes more than source_row: {row['id']}")
        mapping.append({
            "id": row["id"],
            "recorded_source_row": old["provenance"]["source_row"],
            "correct_source_row": row["provenance"]["source_row"],
            "state_sha256": sha256_text(row["state"]),
            "label": row["label"],
        })
    return {
        "schema_version": 1,
        "benchmark_sha256": sha256_file(core_path),
        "source_revision": BANKING77_COMMIT,
        "source_sha256": BANKING77_TEST_SHA256,
        "index_basis": "zero-based CSV data row, excluding the header",
        "scope": "provenance only; frozen inputs, labels and prediction values are unchanged",
        "rows": mapping,
    }


def convert_boolq(path: Path) -> list[dict]:
    source_rows = parquet.read_table(path).to_pylist()
    indexed = [{**source, "source_row": index, "label": "yes" if source["answer"] else "no"}
               for index, source in enumerate(source_rows)]
    selected = _balanced(indexed, "label", ["no", "yes"], 100)
    rows = []
    for source in selected:
        row = {
            "id": f"boolq/validation/{source['source_row']}",
            "dataset": "boolq-balanced",
            "group_id": str(source["source_row"]),
            "state": source["passage"],
            "question": source["question"],
            "options": [
                {"id": "yes", "description": "The passage supports yes."},
                {"id": "no", "description": "The passage supports no."},
            ],
            "label": source["label"],
            "provenance": {
                "source": "google/boolq",
                "revision": BOOLQ_REVISION,
                "source_split": "validation",
                "source_row": source["source_row"],
                "license": "CC-BY-SA-3.0",
                "selection": "first 100 validation rows per answer",
            },
        }
        validate_row(row)
        rows.append(row)
    return rows


def convert_wanli(path: Path) -> list[dict]:
    source_rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    mapping = {"entailment": "supported", "neutral": "insufficient", "contradiction": "contradicted"}
    indexed = [{**source, "label": mapping[source["gold"]]} for source in source_rows]
    selected = _balanced(indexed, "label", ["supported", "insufficient", "contradicted"], 50)
    rows = []
    for source in selected:
        row = {
            "id": f"wanli/test/{source['id']}",
            "dataset": "wanli-balanced",
            "group_id": str(source["pairID"]),
            "state": {"premise": source["premise"], "hypothesis": source["hypothesis"]},
            "question": "What relation does the premise have to the hypothesis?",
            "options": [
                {"id": "supported", "description": "The premise supports the hypothesis."},
                {"id": "insufficient", "description": "The premise does not settle the hypothesis."},
                {"id": "contradicted", "description": "The premise contradicts the hypothesis."},
            ],
            "label": source["label"],
            "provenance": {
                "source": "alisawuffles/WANLI",
                "revision": WANLI_REVISION,
                "source_split": "test",
                "source_id": source["id"],
                "license": "CC-BY-4.0",
                "selection": "first 50 test rows per mapped relation",
            },
        }
        validate_row(row)
        rows.append(row)
    return rows


def convert_massive(path: Path) -> list[dict]:
    source_rows = []
    with tarfile.open(path, "r:gz") as archive:
        for locale in MASSIVE_LOCALES:
            member = archive.getmember(f"1.1/data/{locale}.jsonl")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing MASSIVE locale {locale}")
            for line in stream:
                source = json.loads(line)
                if source["partition"] == "test" and source["intent"] in MASSIVE_INTENTS:
                    source_rows.append(source)
    selected = []
    for locale in MASSIVE_LOCALES:
        locale_rows = [row for row in source_rows if row["locale"] == locale]
        selected.extend(_balanced(locale_rows, "intent", MASSIVE_INTENTS, 2))
    options = [{"id": label, "description": _description(label)} for label in MASSIVE_INTENTS]
    rows = []
    for source in selected:
        row = {
            "id": f"massive/test/{source['locale']}/{source['id']}",
            "dataset": "massive-5-language-12-intent",
            "group_id": source["intent"],
            "state": source["utt"],
            "question": "Which intent best matches this request?",
            "options": copy.deepcopy(options),
            "label": source["intent"],
            "provenance": {
                "source": "AmazonScience/massive",
                "revision": MASSIVE_REVISION,
                "dataset_version": "1.1",
                "source_split": "test",
                "source_id": source["id"],
                "locale": source["locale"],
                "license": "CC-BY-4.0",
                "selection": "first 2 test rows for each frozen intent and locale",
            },
        }
        validate_row(row)
        rows.append(row)
    return rows


def convert_mmlu_pro(path: Path) -> list[dict]:
    source_rows = parquet.read_table(path).to_pylist()
    categories = sorted({row["category"] for row in source_rows})
    selected = _balanced(source_rows, "category", categories, 5)
    rows = []
    for source in selected:
        option_ids = [f"option_{index}" for index in range(len(source["options"]))]
        row = {
            "id": f"mmlu-pro/test/{source['question_id']}",
            "dataset": "mmlu-pro-domain-balanced",
            "group_id": source["category"],
            "state": "Answer using the stated question and options.",
            "question": source["question"],
            "options": [
                {"id": option_id, "description": description}
                for option_id, description in zip(option_ids, source["options"])
            ],
            "label": f"option_{source['answer_index']}",
            "provenance": {
                "source": "TIGER-Lab/MMLU-Pro",
                "revision": MMLU_PRO_REVISION,
                "source_split": "test",
                "source_id": source["question_id"],
                "category": source["category"],
                "license": "MIT",
                "selection": "first 5 test rows per category",
            },
        }
        validate_row(row)
        rows.append(row)
    return rows


def convert_jevbench(files: dict[str, Path]) -> list[dict]:
    rows = []
    for split, path in sorted(files.items()):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            source = json.loads(line)
            labels = [str(label) for label in source["labels"]]
            criteria = source["question"].get("criteria", {})
            descriptions = []
            if isinstance(criteria, list):
                if len(criteria) != len(labels):
                    raise ValueError(f"Criteria count does not match labels for {source['id']}")
                descriptions = criteria
            else:
                for label in labels:
                    key = {"yes": "true", "no": "false"}.get(label, label)
                    descriptions.append(criteria.get(key, label.replace("_", " ")))
            row = {
                "id": f"jevbench/{split}/{source['id']}",
                "dataset": f"jevbench-{split}",
                "group_id": source.get("group") or source["id"],
                "state": source["state"],
                "question": source["question"]["instructions"],
                "options": [
                    {"id": label, "description": description}
                    for label, description in zip(labels, descriptions)
                ],
                "label": str(source["expected"]),
                "provenance": {
                    "source": "fstandhartinger/jevbench",
                    "revision": JEVBENCH_REVISION,
                    "source_id": source["id"],
                    "source_split": split,
                    "family": source.get("family"),
                    "license": "MIT",
                },
            }
            validate_row(row)
            rows.append(row)
    if len(rows) != 231:
        raise ValueError(f"Expected 231 public JevBench rows, received {len(rows)}")
    return rows


def read_owned_base(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 36 or len({row["id"] for row in rows}) != 36:
        raise ValueError("Owned base must contain exactly 36 unique cases")
    return rows


def build_owned(path: Path) -> list[dict]:
    output = []
    for base in read_owned_base(path):
        common = {
            "dataset": "owned-robustness",
            "group_id": base["id"],
            "question": base["question"],
            "state": base["state"],
            "options": copy.deepcopy(base["options"]),
            "label": base["label"],
        }
        variants = [
            ("original", {}),
            ("option_reversal", {"options": list(reversed(common["options"]))}),
            ("question_reworded", {"question": base["reworded_question"]}),
            ("irrelevant_context", {"state": base["state"] + "\n\nIrrelevant note: " + base["irrelevant_context"]}),
            ("missing_evidence", {"state": base["missing_state"], "label": "insufficient"}),
        ]
        for variant, changes in variants:
            row = {
                **common,
                **changes,
                "id": f"owned/{base['id']}/{variant}",
                "provenance": {
                    "source": "project-authored",
                    "family": base["family"],
                    "variant": variant,
                    "parent_id": base["id"],
                    "rationale": (
                        base["missing_rationale"] if variant == "missing_evidence" else base["rationale"]
                    ),
                    "evidence_span": (
                        base["missing_evidence_span"] if variant == "missing_evidence" else base["evidence_span"]
                    ),
                    "review_status": base["review_status"],
                    "license": "MIT",
                },
            }
            validate_row(row)
            output.append(row)
    if len(output) != 180:
        raise ValueError("Owned suite must contain 180 rows")
    return output


def build_core(cache: Path, owned_base: Path, output: Path) -> dict:
    jevbench = convert_jevbench(fetch_jevbench(cache))
    sources = fetch_public_sources(cache)
    banking77 = convert_banking77(sources["banking77"])
    boolq = convert_boolq(sources["boolq"])
    wanli = convert_wanli(sources["wanli"])
    massive = convert_massive(sources["massive"])
    mmlu_pro = convert_mmlu_pro(sources["mmlu_pro"])
    owned = build_owned(owned_base)
    rows = jevbench + banking77 + boolq + wanli + massive + mmlu_pro + owned
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Core suite contains duplicate row ids")
    write_jsonl_create(output, rows)
    return {
        "rows": len(rows),
        "sources": {
            "jevbench": len(jevbench),
            "banking77": len(banking77),
            "boolq": len(boolq),
            "wanli": len(wanli),
            "massive": len(massive),
            "mmlu_pro": len(mmlu_pro),
            "owned": len(owned),
        },
        "output": str(output),
    }


def build_smoke(core_path: Path, output: Path) -> dict:
    rows = [
        json.loads(line)
        for line in core_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selectors = [
        ("jevbench", lambda row: row["dataset"] == "jevbench-easy"),
        ("banking77", lambda row: row["dataset"] == "banking77-12-intent"),
        ("boolq", lambda row: row["dataset"] == "boolq-balanced"),
        ("wanli", lambda row: row["dataset"] == "wanli-balanced"),
        ("massive", lambda row: row["dataset"] == "massive-5-language-12-intent"),
        ("mmlu_pro", lambda row: row["dataset"] == "mmlu-pro-domain-balanced"),
        (
            "owned",
            lambda row: (
                row["dataset"] == "owned-robustness"
                and row.get("provenance", {}).get("variant") == "original"
            ),
        ),
    ]
    selected = []
    for source, predicate in selectors:
        try:
            selected.append(next(row for row in rows if predicate(row)))
        except StopIteration as error:
            raise ValueError(f"Core suite has no smoke row for {source}") from error
    write_jsonl_create(output, selected)
    return {"rows": len(selected), "output": str(output)}


def build_jevbench_pilot(core_path: Path, output: Path, per_tier: int = 10) -> dict:
    rows = [
        json.loads(line)
        for line in core_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = []
    counts = {}
    for tier in ("easy", "original", "hard"):
        tier_rows = [row for row in rows if row["dataset"] == f"jevbench-{tier}"]
        if len(tier_rows) < per_tier:
            raise ValueError(f"Need {per_tier} JevBench {tier} rows")
        chosen = tier_rows[:per_tier]
        selected.extend(chosen)
        counts[tier] = len(chosen)
    write_jsonl_create(output, selected)
    return {"rows": len(selected), "tiers": counts, "output": str(output)}

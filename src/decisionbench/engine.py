from __future__ import annotations

import math
import os
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import platform
import threading
import time
import warnings

import numpy
import psutil

from .prompt import LABELS, PROMPT_VERSION, SYSTEM_PROMPT, messages, prompt_record, render_plain
from .schema import canonical_json, sha256_file, sha256_text, validate_row


def softmax(values: list[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite logits")
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def logsumexp(values: numpy.ndarray) -> float:
    peak = float(values.max())
    return peak + float(numpy.log(numpy.exp(values - peak).sum()))


class CpuEngine:
    """Small llama.cpp adapter that returns final-position vocabulary logits."""

    def __init__(
        self,
        model_path: Path,
        *,
        model_id: str,
        threads: int | None = None,
        context_tokens: int = 4096,
        chat_format: str | None = None,
        benchmark_date: str = "2026-09-24",
    ):
        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise RuntimeError("Install the CPU extra: pip install -e '.[cpu]'") from error
        if not model_path.is_file():
            raise ValueError(f"GGUF model not found: {model_path}")
        if context_tokens < 1:
            raise ValueError("context_tokens must be positive")
        if threads is not None and threads < 1:
            raise ValueError("threads must be positive")
        self.model_path = model_path
        self.model_id = model_id
        self.threads = threads or os.cpu_count() or 4
        self.context_tokens = context_tokens
        self.benchmark_date = benchmark_date
        load_started = time.perf_counter()
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=context_tokens,
            n_threads=self.threads,
            n_threads_batch=self.threads,
            n_gpu_layers=0,
            logits_all=True,
            chat_format=chat_format,
            verbose=False,
        )
        load_seconds = time.perf_counter() - load_started
        self._formatter = self._build_formatter()
        from llama_cpp import llama_print_system_info

        source_dir = Path(__file__).parent
        self.metadata = {
            "id": model_id,
            "gguf_file": model_path.name,
            "gguf_bytes": model_path.stat().st_size,
            "gguf_sha256": sha256_file(model_path),
            "threads": self.threads,
            "context_tokens": context_tokens,
            "n_gpu_layers": 0,
            "chat_format": self._llm.chat_format,
            "prompt_rendering": "gguf-template" if self._formatter else "plain-text-fallback",
            "benchmark_date": benchmark_date,
            "load_seconds": load_seconds,
            "process_rss_after_load_bytes": psutil.Process().memory_info().rss,
            "prompt_version": PROMPT_VERSION,
            "instruction_sha256": sha256_text(SYSTEM_PROMPT),
            "implementation_sha256": sha256_text(canonical_json({
                name: sha256_text((source_dir / name).read_text(encoding="utf-8"))
                for name in ("engine.py", "prompt.py", "schema.py")
            })),
        }
        self.environment = {
            "python": platform.python_version(),
            "system": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "total_memory_bytes": psutil.virtual_memory().total,
            "packages": {
                name: version(name)
                for name in ("llama-cpp-python", "numpy", "psutil", "Jinja2")
            },
            "llama_cpp_build": llama_print_system_info().decode("utf-8"),
        }

    def close(self) -> None:
        self._llm.close()

    def _special_token_text(self, token_id: int) -> str:
        if token_id < 0:
            return ""
        return self._llm.detokenize([token_id], special=True).decode("utf-8", errors="replace")

    def _build_formatter(self):
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        metadata = self._llm.metadata
        chat_format = self._llm.chat_format
        template = None
        if chat_format:
            template = metadata.get(f"tokenizer.chat_template.{chat_format}")
        template = template or metadata.get("tokenizer.chat_template")
        if not template:
            warnings.warn("GGUF has no chat template; using the recorded plain-text fallback.", stacklevel=2)
            return None
        formatter = Jinja2ChatFormatter(
            template=template,
            eos_token=self._special_token_text(self._llm.token_eos()),
            bos_token=self._special_token_text(self._llm.token_bos()),
            add_generation_prompt=True,
        )
        frozen = datetime.strptime(self.benchmark_date, "%Y-%m-%d")
        formatter.strftime_now = frozen.strftime
        return formatter

    def _render(self, row: dict) -> tuple[str, bool]:
        if self._formatter is None:
            return render_plain(row), True
        rendered = self._formatter(messages=messages(row), enable_thinking=False)
        return rendered.prompt, not rendered.added_special

    def _tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)

    def _alias_ids(self, prompt: str, count: int, *, add_bos: bool) -> list[int]:
        prefix = self._tokenize(prompt, add_bos=add_bos)
        aliases = []
        for label in LABELS[:count]:
            extended = self._tokenize(prompt + label, add_bos=add_bos)
            if len(extended) != len(prefix) + 1 or extended[:-1] != prefix:
                raise ValueError(f"Alias {label!r} is not one token at the answer boundary")
            aliases.append(extended[-1])
        if len(aliases) != len(set(aliases)):
            raise ValueError("Alias token ids collide")
        return aliases

    def inspect_aliases(self, row: dict) -> dict:
        validate_row(row)
        prompt, add_bos = self._render(row)
        return {
            "prompt": prompt_record(prompt),
            "alias_token_ids": self._alias_ids(prompt, len(row["options"]), add_bos=add_bos),
        }

    def score(self, row: dict) -> dict:
        validate_row(row)
        started = time.perf_counter()
        prompt, add_bos = self._render(row)
        tokens = self._tokenize(prompt, add_bos=add_bos)
        if not tokens or len(tokens) > self.context_tokens:
            raise ValueError(f"Prompt uses {len(tokens)} tokens; limit is {self.context_tokens}")
        aliases = self._alias_ids(prompt, len(row["options"]), add_bos=add_bos)
        self._llm.reset()
        process = psutil.Process()
        peak_rss = process.memory_info().rss
        stop_sampling = threading.Event()

        def sample_memory() -> None:
            nonlocal peak_rss
            while not stop_sampling.wait(0.01):
                peak_rss = max(peak_rss, process.memory_info().rss)

        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        mark = time.perf_counter()
        try:
            self._llm.eval(tokens)
            elapsed = time.perf_counter() - mark
        finally:
            stop_sampling.set()
            sampler.join()
            peak_rss = max(peak_rss, process.memory_info().rss)
        scores = numpy.asarray(self._llm.scores)
        row_index = self._llm.n_tokens - 1
        if scores.ndim != 2 or row_index < 0 or scores.shape[0] <= row_index:
            raise RuntimeError("llama.cpp returned no token score matrix")
        vocabulary = scores[row_index].astype(numpy.float64)
        selected = vocabulary[aliases].tolist()
        probabilities = softmax(selected)
        choice = int(numpy.argmax(probabilities))
        return {
            "id": row["id"],
            "dataset": row["dataset"],
            "option_ids": [option["id"] for option in row["options"]],
            "alias_token_ids": aliases,
            "option_logits": selected,
            "probabilities": probabilities,
            "prediction_id": row["options"][choice]["id"],
            "input_tokens": len(tokens),
            "allowed_token_mass": float(math.exp(logsumexp(numpy.asarray(selected)) - logsumexp(vocabulary))),
            "full_vocab_argmax_id": int(vocabulary.argmax()),
            "full_vocab_argmax_text": self._special_token_text(int(vocabulary.argmax())),
            "forward_seconds": elapsed,
            "total_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "prompt": prompt_record(prompt),
            "model": self.metadata,
            "probability_status": "conditional on the declared aliases; not calibrated confidence",
        }

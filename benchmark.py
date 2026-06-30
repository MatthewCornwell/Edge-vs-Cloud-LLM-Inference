#!/usr/bin/env python3
"""
LLM benchmark runner with optional energy logging.

Backends: llama.cpp (local gguf), rkllama (rk3588 NPU), gemini.
Energy: pynvml for nvidia gpus, tapo p110 smart plug for whole system
"""

import argparse
import asyncio
import csv
import gc
import json
import logging
import os
import random
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import dotenv
import faiss
import numpy as np
import pynvml
import requests
from datasets import load_dataset
from google import genai
from google.genai import types as genai_types
from sentence_transformers import SentenceTransformer
from tapo import ApiClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

RESULT_FIELDS = [
    "timestamp", "label", "backend", "model", "mode",
    "prompt_id", "run_idx", "prompt_text", "response_text",
    "input_tokens", "output_tokens", "ttft_s", "total_time_s",
    "prefill_tps", "decode_tps",
    "total_energy_j", "avg_power_w", "peak_power_w",
    "energy_per_token_j", "hit_max_tokens", "max_tokens_cap", "notes",
]


# --- energy monitoring ---

class PynvmlEnergyMonitor:
    method = "pynvml"

    def __init__(self, gpu_index: int = 0, poll_hz: float = 10.0):
        self.poll_hz = poll_hz
        self.poll_interval = 1.0 / poll_hz
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._samples: List[float] = []
        self._times: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error_logged = False

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self._samples.append(w)
                self._times.append(time.time())
            except Exception as e:
                if not self._error_logged:
                    logger.error(f"Energy polling error (suppressing further): {e}")
                    self._error_logged = True
            time.sleep(self.poll_interval)

    def start(self) -> None:
        self._samples.clear()
        self._times.clear()
        self._stop.clear()
        self._error_logged = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        n = len(self._samples)
        if n < 2:
            avg = self._samples[0] if self._samples else 0.0
            return {"total_energy_j": 0.0, "avg_power_w": avg, "peak_power_w": avg, "n_samples": n}

        energy = sum(
            (self._samples[i] + self._samples[i - 1]) / 2.0 * (self._times[i] - self._times[i - 1])
            for i in range(1, n)
        )
        return {
            "total_energy_j": energy,
            "avg_power_w": sum(self._samples) / n,
            "peak_power_w": max(self._samples),
            "n_samples": n,
        }


class TapoEnergyMonitor(PynvmlEnergyMonitor):
    method = "tapo"

    def __init__(self, ip: str, email: str, password: str, poll_hz: float = 1.0):
        # new __init__ for tapoenergymonitor
        self.poll_hz = poll_hz
        self.poll_interval = 1.0 / poll_hz
        self._samples: List[float] = []
        self._times: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error_logged = False

        # tapo's lib is async, this fix is discussed in the report
        if sys.platform == "win32":
            self._asyncio_loop = asyncio.SelectorEventLoop()
        else:
            self._asyncio_loop = asyncio.new_event_loop()

        client = ApiClient(email, password)
        self._device = self._asyncio_loop.run_until_complete(client.p110(ip))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._asyncio_loop.run_until_complete(self._device.get_current_power())
                w = float(data.current_power)
                self._samples.append(w)
                self._times.append(time.time())
            except Exception as e:
                if not self._error_logged:
                    logger.error(f"Energy polling error (suppressing further): {e}")
                    self._error_logged = True
            time.sleep(self.poll_interval)


# --- backends ---

class LlamaCppBackend:
    name = "llama_cpp"

    def __init__(self, args):
        self.args = args
        from llama_cpp import Llama
        if not args.model_path or not os.path.exists(args.model_path):
            raise SystemExit(f"--model-path missing or not found: {args.model_path}")
        self.llm = Llama(
            model_path=args.model_path,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.ctx,
            verbose=False,
        )
        self._model_name = Path(args.model_path).stem

    def model_id(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> Dict[str, Any]:
        if "</think>" not in prompt:
            prompt = prompt.rstrip() + "\n</think>\n\n"

        prompt_tokens = self.llm.tokenize(prompt.encode("utf-8"))
        n_input = len(prompt_tokens)
        t0 = time.perf_counter()
        ttft = None
        chunks = []
        n_output = 0
        finish_reason = None

        for chunk in self.llm.create_completion(
            prompt=prompt,
            max_tokens=self.args.max_tokens,
            stream=True,
            temperature=self.args.temperature,
        ):
            text = chunk["choices"][0]["text"]
            if "<think>" in text or "</think>" in text:
                text = text.replace("<think>", "").replace("</think>", "")
                if not text:
                    continue
            if ttft is None and text:
                ttft = time.perf_counter() - t0
            chunks.append(text)
            n_output += 1
            fr = chunk["choices"][0].get("finish_reason")
            if fr is not None:
                finish_reason = fr
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens"):
                    n_output = usage["completion_tokens"]

        total = time.perf_counter() - t0
        if ttft is None:
            ttft = total
        decode_t = total - ttft
        if decode_t <= 0:
            decode_tps = n_output / total if total > 0 else 0.0
        else:
            decode_tps = n_output / decode_t
            # sanity check: llama.cpp can lie about timings on tiny outputs
            if decode_tps > 500:
                decode_tps = n_output / total if total > 0 else 0.0

        prefill_tps = n_input / ttft if ttft > 0 else 0.0

        return {
            "response": "".join(chunks),
            "input_tokens": n_input,
            "output_tokens": n_output,
            "ttft_s": ttft,
            "total_time_s": total,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "hit_max_tokens": (finish_reason == "length") or (n_output >= self.args.max_tokens),
            "notes": "",
        }


class RkllamaBackend:
    name = "rkllama"

    def __init__(self, args):
        self.args = args
        self.host = args.rkllama_host.rstrip("/")
        self.model = args.model

    def model_id(self) -> str:
        return self.model or "unknown"

    def generate(self, prompt: str) -> Dict[str, Any]:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model, "prompt": prompt, "stream": True,
            "options": {
                "temperature": self.args.temperature,
                "num_predict": self.args.max_tokens,
            },
        }
        t0 = time.perf_counter()
        ttft = None
        chunks = []
        last = {}
        notes = []

        with requests.post(url, json=payload, stream=True, timeout=900) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("response"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks.append(obj["response"])
                if obj.get("done"):
                    last = obj
                    break

        total = time.perf_counter() - t0
        text = "".join(chunks)
        if ttft is None:
            ttft = total

        n_input = int(last.get("prompt_eval_count", 0))
        n_output = int(last.get("eval_count", 0))

        if n_input == 0:
            n_input = max(1, len(prompt) // 4)
            notes.append("input_tokens estimated from char count")
        if n_output == 0:
            n_output = max(1, len(text) // 4)
            notes.append("output_tokens estimated from char count")

        # rkllama reports eval_duration in nanoseconds
        eval_raw = float(last.get("eval_duration", 0))
        prompt_eval_raw = float(last.get("prompt_eval_duration", 0))

        if eval_raw > 0:
            decode_tps = n_output / (eval_raw / 1e9)
        elif total > ttft:
            decode_tps = n_output / (total - ttft)
        else:
            decode_tps = 0.0

        prefill_tps = (n_input / (prompt_eval_raw / 1e9)) if prompt_eval_raw > 0 else 0.0

        if decode_tps > 500 and total > ttft:
            decode_tps = n_output / (total - ttft)
            notes.append("decode_tps recomputed from wall clock")

        return {
            "response": text,
            "input_tokens": n_input,
            "output_tokens": n_output,
            "ttft_s": ttft,
            "total_time_s": total,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "hit_max_tokens": (last.get("done_reason") == "length") or (n_output >= self.args.max_tokens),
            "notes": "; ".join(notes),
        }


class GeminiBackend:
    name = "gemini"

    def __init__(self, args):
        self.args = args
        api_key = os.environ.get("GEMINI_API_KEY") or args.gemini_api_key
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY env var or pass --gemini-api-key")
        self.client = genai.Client(api_key=api_key)
        self._model_name = args.model
        self._rate_limit_sleep = args.gemini_rate_limit_sleep

    def model_id(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        ttft = None
        chunks = []
        notes = []
        n_input, n_output = 0, 0
        finish_reason = None

        for chunk in self.client.models.generate_content_stream(
            model=self._model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=self.args.temperature,
                max_output_tokens=self.args.max_tokens,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0), #Critical, no thinking budget to align with other backends
            ),
        ):
            if chunk.text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(chunk.text)
            if chunk.usage_metadata:
                um = chunk.usage_metadata
                if um.prompt_token_count:
                    n_input = um.prompt_token_count
                if um.candidates_token_count:
                    n_output = um.candidates_token_count
            if chunk.candidates:
                fr = chunk.candidates[0].finish_reason
                if fr:
                    finish_reason = fr

        total = time.perf_counter() - t0
        text = "".join(chunks)

        if n_input == 0:
            n_input = max(1, len(prompt) // 4)
            notes.append("input_tokens estimated from char count")
        if n_output == 0:
            n_output = max(1, len(text) // 4)
            notes.append("output_tokens estimated from char count")
        if ttft is None:
            ttft = total

        if total > ttft:
            decode_tps = n_output / (total - ttft)
            if decode_tps > 500 and total > 0:
                decode_tps = n_output / total
        elif total > 0:
            decode_tps = n_output / total
        else:
            decode_tps = 0.0

        if self._rate_limit_sleep > 0:
            time.sleep(self._rate_limit_sleep)

        return {
            "response": text,
            "input_tokens": n_input,
            "output_tokens": n_output,
            "ttft_s": ttft,
            "total_time_s": total,
            "prefill_tps": 0.0,  
            "decode_tps": decode_tps,
            "hit_max_tokens": str(finish_reason).endswith("MAX_TOKENS") or n_output >= self.args.max_tokens,
            "notes": "; ".join(notes),
        }


# --- RAG ---

class RAGEngine:
    CHUNK_PREFIX = "rag_chunk_"

    def __init__(self, chunks_dir: str = "rag_chunks", embed_model: str = "all-MiniLM-L6-v2"):
        self.chunks_dir = Path(chunks_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model_name = embed_model
        self._embedder = None
        self.index = None
        self.all_docs: List[str] = []

    def _get_embedder(self):
        if self._embedder is None:
            logger.info(f"Loading embedder ({self.embed_model_name}, CPU)...")
            self._embedder = SentenceTransformer(self.embed_model_name, device="cpu")
        return self._embedder

    def _chunk_files(self) -> List[Path]:
        files = list(self.chunks_dir.glob(f"{self.CHUNK_PREFIX}*.npz"))
        files.sort(key=lambda p: int(re.search(r"_(\d+)\.npz", p.name).group(1)))
        return files

    def index_build(self, num_docs: Optional[int] = None, batch_size: int = 500,
                    dataset: str = "wikimedia/wikipedia", config: str = "20231101.simple"):
        embedder = self._get_embedder()
        existing = self._chunk_files()
        existing_ids = [int(re.search(r"_(\d+)\.npz", p.name).group(1)) for p in existing]
        next_id = max(existing_ids) + 1 if existing_ids else 0

        already_done = 0
        for p in existing:
            try:
                with np.load(p) as data:
                    already_done += len(data["docs"])
            except Exception:
                already_done += 500

        if num_docs and already_done >= num_docs:
            logger.info(f"Index already has ~{already_done} docs. Skipping build.")
            return

        logger.info(f"Streaming dataset {dataset}/{config} ...")
        ds = load_dataset(dataset, config, split="train", streaming=True)

        batch_docs: List[str] = []
        progress = already_done
        t_start = time.time()

        for i, item in enumerate(ds):
            if i < already_done:
                continue
            if num_docs and progress >= num_docs:
                break
            text = item.get("text", "")[:800]
            if len(text) <= 100:
                continue
            batch_docs.append(text)
            progress += 1
            if len(batch_docs) >= batch_size:
                embs = embedder.encode(batch_docs, convert_to_numpy=True, normalize_embeddings=True)
                np.savez_compressed(
                    self.chunks_dir / f"{self.CHUNK_PREFIX}{next_id}.npz",
                    docs=np.array(batch_docs), embs=embs.astype("float32")
                )
                next_id += 1
                elapsed = time.time() - t_start
                rate = progress / elapsed if elapsed > 0 else 0.0
                logger.info(f"{progress:,} docs indexed ({rate:.1f} docs/s)")
                batch_docs = []

        if batch_docs:
            embs = embedder.encode(batch_docs, convert_to_numpy=True, normalize_embeddings=True)
            np.savez_compressed(
                self.chunks_dir / f"{self.CHUNK_PREFIX}{next_id}.npz",
                docs=np.array(batch_docs), embs=embs.astype("float32")
            )
        logger.info(f"Index build complete. Total docs: {progress:,}")

    def load(self) -> None:
        files = self._chunk_files()
        if not files:
            raise RuntimeError(f"No chunks found in {self.chunks_dir}.")
        all_embs = []
        for f in files:
            try:
                with np.load(f) as data:
                    self.all_docs.extend(data["docs"])
                    all_embs.append(data["embs"])
            except Exception as e:
                logger.warning(f"Skipping bad chunk {f}: {e}")
        embs = np.vstack(all_embs).astype("float32")
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        logger.info(f"Loaded {len(self.all_docs):,} documents, index dim={embs.shape[1]}")
        gc.collect()

    def query(self, q: str, top_k: int = 2) -> str:
        if self.index is None or self.index.ntotal == 0:
            return ""
        embedder = self._get_embedder()
        q_emb = embedder.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, idxs = self.index.search(q_emb, top_k)
        passages = [f"- {self.all_docs[i][-600:]}" for i in idxs[0] if 0 <= i < len(self.all_docs)]
        return "\n".join(passages)


def build_ifeval_subset(n: int, seed: int, out_path: str) -> None:
    logger.info("Loading google/IFEval ...")
    ds = load_dataset("google/IFEval", split="train")
    by_cat: Dict[str, List[Dict]] = {}
    for row in ds:
        cats = row.get("instruction_id_list", [])
        primary = cats[0].split(":")[0] if cats else "unknown"
        by_cat.setdefault(primary, []).append(row)

    rng = random.Random(seed)
    cats = sorted(by_cat.keys())
    per_cat = max(1, n // len(cats))
    picked: List[Dict] = []
    for c in cats:
        items = by_cat[c][:]
        rng.shuffle(items)
        picked.extend(items[:per_cat])
    rng.shuffle(picked)
    if len(picked) < n:
        picked_set = set(map(id, picked))
        leftover = [r for c in cats for r in by_cat[c] if id(r) not in picked_set]
        rng.shuffle(leftover)
        picked.extend(leftover[: n - len(picked)])
    picked = picked[:n]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({
                "prompt_id": str(r["key"]),
                "prompt": r["prompt"],
                "instruction_id_list": r.get("instruction_id_list", []),
                "kwargs": r.get("kwargs", []),
            }) + "\n")
    logger.info(f"Wrote {len(picked)} prompts -> {out_path}")


def load_prompts(path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pid = obj.get("prompt_id") or obj.get("key") or len(rows)
                rows.append({"prompt_id": str(pid), "prompt": obj["prompt"]})
        else:
            data = json.load(f)
            for i, obj in enumerate(data):
                pid = obj.get("prompt_id") or obj.get("key") or i
                rows.append({"prompt_id": str(pid), "prompt": obj["prompt"]})
    return rows[:limit] if limit else rows


def run_benchmark(args, backend, rag: Optional[RAGEngine], prompts: List[Dict[str, str]]) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        try:
            with open(out_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("label") == args.label and row.get("response_text"):
                        done.add((row.get("mode", ""), row.get("prompt_id", ""), int(row.get("run_idx") or 0)))
        except Exception as e:
            logger.warning(f"Could not read existing CSV for resume tracking: {e}")

    # write the header only if starting a fresh file
    write_header = True
    if out_path.exists() and out_path.stat().st_size > 0:
        write_header = False

    # newline="" is needed on Windows or csv emits blank lines between rows
    f_out = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=RESULT_FIELDS)
    if write_header:
        writer.writeheader()

    modes = []
    if args.rag_mode in ("raw", "both"):
        modes.append("raw")
    if args.rag_mode in ("rag", "both"):
        modes.append("rag")
    if "rag" in modes and rag is None:
        raise SystemExit("--rag-mode requires RAG enabled.")

    # Energy monitor
    if args.energy == "pynvml":
        monitor = PynvmlEnergyMonitor(gpu_index=args.gpu_index, poll_hz=args.energy_poll_hz or 10.0)
    elif args.energy == "tapo":
        if not all([args.tapo_ip, args.tapo_email, args.tapo_password]):
            raise SystemExit("tapo energy method requires TAPO_IP, TAPO_EMAIL, TAPO_PASSWORD (env or args)")
        monitor = TapoEnergyMonitor(args.tapo_ip, args.tapo_email, args.tapo_password, poll_hz=args.energy_poll_hz or 1.0)
    else:
        monitor = None

    # Warmup by using the first prompt from the benchmark set
    if args.warmup and backend.name != "gemini":
        logger.info("Running warmup pass...")
        try:
            backend.generate(prompts[0]["prompt"])
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    n_total = len(prompts) * len(modes) * args.runs
    n_done, n_skip = 0, 0
    t_start = time.time()

    for prompt_obj in prompts:
        pid = prompt_obj["prompt_id"]
        ptext = prompt_obj["prompt"]

        for mode in modes:
            for run_idx in range(args.runs):
                if (mode, pid, run_idx) in done:
                    n_skip += 1
                    n_done += 1
                    continue

                full_prompt = ptext
                if mode == "rag" and rag:
                    ctx = rag.query(ptext, top_k=args.rag_top_k)
                    if ctx:
                        full_prompt = (
                            f"Use context to answer. If irrelevant, ignore.\n"
                            f"Context:\n{ctx}\n\nInstruction: {ptext}\nAnswer:"
                        )

                notes = []
                res = None
                if monitor is None:
                    e_stats = {"total_energy_j": "N/A", "avg_power_w": "N/A", "peak_power_w": "N/A", "n_samples": 0}
                else:
                    e_stats = {"total_energy_j": 0.0, "avg_power_w": 0.0, "peak_power_w": 0.0, "n_samples": 0}

                try:
                    if monitor is not None:
                        monitor.start()
                    try:
                        res = backend.generate(full_prompt)
                        if res.get("notes"):
                            notes.append(res["notes"])
                    finally:
                        if monitor is not None:
                            try:
                                e_stats = monitor.stop()
                            except Exception as ex:
                                notes.append(f"monitor_stop_error: {type(ex).__name__}")
                except Exception as e:
                    notes.append(f"inference_error: {type(e).__name__}:{e}")

                if res is None:
                    res = {
                        "response": "", "input_tokens": 0, "output_tokens": 0,
                        "ttft_s": 0.0, "total_time_s": 0.0,
                        "prefill_tps": 0.0, "decode_tps": 0.0,
                        "hit_max_tokens": False,
                    }

                n_out = res.get("output_tokens", 0)
                if monitor is None:
                    e_per_tok = "N/A"
                else:
                    e_per_tok = (e_stats["total_energy_j"] / n_out) if n_out > 0 else 0.0

                if args.energy in ("pynvml", "tapo") and e_stats.get("n_samples", 0) < 5:
                    notes.append(f"low integration confidence ({e_stats.get('n_samples')} samples)")

                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "label": args.label,
                    "backend": backend.name,
                    "model": backend.model_id(),
                    "mode": mode,
                    "prompt_id": pid,
                    "run_idx": run_idx,
                    "prompt_text": ptext,
                    "response_text": res.get("response", ""),
                    "input_tokens": res.get("input_tokens", 0),
                    "output_tokens": n_out,
                    "ttft_s": res.get("ttft_s", 0.0),
                    "total_time_s": res.get("total_time_s", 0.0),
                    "prefill_tps": res.get("prefill_tps", 0.0),
                    "decode_tps": res.get("decode_tps", 0.0),
                    "total_energy_j": e_stats["total_energy_j"],
                    "avg_power_w": e_stats["avg_power_w"],
                    "peak_power_w": e_stats["peak_power_w"],
                    "energy_per_token_j": e_per_tok,
                    "hit_max_tokens": bool(res.get("hit_max_tokens", False)),
                    "max_tokens_cap": args.max_tokens,
                    "notes": "; ".join(notes),
                }

                try:
                    writer.writerow(row)
                    f_out.flush()
                except Exception as ex:
                    logger.error(f"Failed to write row pid={pid}: {ex}")

                n_done += 1
                elapsed = time.time() - t_start
                completed = n_done - n_skip
                if elapsed > 0 and completed > 0:
                    rate = completed / elapsed
                    remaining = (n_total - n_done) / rate
                else:
                    remaining = 0
                cap_flag = "*" if row["hit_max_tokens"] else " "
                e_total_str = f"{row['total_energy_j']:.1f}J" if isinstance(row['total_energy_j'], (int, float)) else "N/A"
                e_per_tok_str = f"{e_per_tok:.3f}" if isinstance(e_per_tok, (int, float)) else "N/A"
                note_str = f" NOTE:{'; '.join(notes)}" if notes else ""
                logger.info(
                    f"[{n_done}/{n_total}] mode={mode} pid={pid} run={run_idx} "
                    f"out_tok={n_out}{cap_flag} tps={row['decode_tps']:.1f} "
                    f"ttft={row['ttft_s']:.2f}s E={e_total_str} "
                    f"J/tok={e_per_tok_str} ETA={remaining / 60:.0f}m"
                    + note_str
                )

    f_out.close()
    logger.info(f"Done. Results written to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-platform LLM inference benchmark.")

    # Utilities
    p.add_argument("--build-ifeval", type=str, metavar="OUT", help="Build IFEval subset and exit.")
    p.add_argument("--ifeval-n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--build-rag-index", action="store_true")
    p.add_argument("--rag-num-docs", type=int, default=None)
    p.add_argument("--rag-chunks-dir", default="rag_chunks")

    # Run config
    p.add_argument("--backend", choices=["llama_cpp", "rkllama", "gemini"])
    p.add_argument("--label", required=False)
    p.add_argument("--prompts")
    p.add_argument("--out", default="data/results.csv")
    p.add_argument("--limit", type=int)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--rag-mode", choices=["raw", "rag", "both"], default="raw")
    p.add_argument("--rag-top-k", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--warmup", action="store_true", default=True)

    # Generation
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0)

    # Backend specific
    p.add_argument("--model-path")
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--rkllama-host", default="http://localhost:8080")
    p.add_argument("--model")
    p.add_argument("--gemini-api-key")
    p.add_argument("--gemini-rate-limit-sleep", type=float, default=13.0)  

    # Energy
    p.add_argument("--energy", default="none", choices=["none", "pynvml", "tapo"])
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--energy-poll-hz", type=float)
    p.add_argument("--tapo-ip", default=os.environ.get("TAPO_IP"))
    p.add_argument("--tapo-email", default=os.environ.get("TAPO_EMAIL"))
    p.add_argument("--tapo-password", default=os.environ.get("TAPO_PASSWORD"))

    args = p.parse_args()

    if args.build_ifeval:
        build_ifeval_subset(args.ifeval_n, args.seed, args.build_ifeval)
        return
    if args.build_rag_index:
        rag = RAGEngine(chunks_dir=args.rag_chunks_dir)
        rag.index_build(num_docs=args.rag_num_docs)
        return

    if not args.backend or not args.label or not args.prompts:
        raise SystemExit("Missing required arguments: --backend, --label, --prompts")

    backends = {"llama_cpp": LlamaCppBackend, "rkllama": RkllamaBackend, "gemini": GeminiBackend}
    backend = backends[args.backend](args)

    rag = None
    if args.rag_mode in ("rag", "both"):
        rag = RAGEngine(chunks_dir=args.rag_chunks_dir)
        rag.load()

    prompts = load_prompts(args.prompts, limit=args.limit)
    logger.info(
        f"Starting: backend={backend.name} model={backend.model_id()} label={args.label} "
        f"prompts={len(prompts)} modes={args.rag_mode} runs={args.runs} energy={args.energy}"
    )

    run_benchmark(args, backend, rag, prompts)


if __name__ == "__main__":
    main()
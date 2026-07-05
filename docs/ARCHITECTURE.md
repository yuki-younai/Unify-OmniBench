# Unify-OmniBench 架构与接口规范

> 配合 `DESIGN.md` 阅读。本文档提供具体接口签名、关键算法与文件级骨架，便于直接进入实现。

---

## 1. 核心接口签名

### 1.1 Dataset Adapter

```python
# unify_omnibench/datasets/base.py
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any
from ..core.types import Sample

class BaseDatasetAdapter(ABC):
    name: str = ""                  # 注册名

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    # 通用工具：构造 uid
    def make_uid(self, *parts) -> str:
        return f"{self.name}:" + ":".join(str(p) for p in parts)
```

### 1.2 Model

```python
# unify_omnibench/models/base.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from ..core.types import InferenceRequest, Modality

class BaseModel(ABC):
    name: str = ""
    supports_modalities: Tuple[Modality, ...] = ()
    is_thread_safe: bool = False
    supports_batch: bool = False

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def load(self) -> None: ...
    def close(self) -> None: ...

    @abstractmethod
    def generate(self, req: InferenceRequest) -> str: ...

    def generate_batch(self, reqs: List[InferenceRequest]) -> List[str]:
        return [self.generate(r) for r in reqs]
```

### 1.3 Registry

```python
# unify_omnibench/core/registry.py
DATASET_REGISTRY: dict[str, type] = {}
MODEL_REGISTRY:   dict[str, type] = {}

def register_dataset(name):
    def deco(cls):
        cls.name = name
        DATASET_REGISTRY[name] = cls
        return cls
    return deco

def register_model(name):
    def deco(cls):
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls
    return deco

def build_dataset(cfg): return DATASET_REGISTRY[cfg["name"]](cfg)
def build_model(cfg):   return MODEL_REGISTRY[cfg["name"]](cfg)
```

---

## 2. 答案解析（兼容三 Benchmark 的级联策略）

```python
# unify_omnibench/eval/parser.py
import re
from typing import Optional, Dict

LETTERS = ("A", "B", "C", "D")

def extract_choice_letter(text: str, index2ans: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    级联策略（按优先级）：
      1) JSON: {"answer":"X"}           （OmniVideoBench）
      2) \\boxed{X} / \\boxed{\\text{X}} （OmniVideoBench）
      3) 首字符为 A-D                    （Daily-Omni）
      4) 文本中第一个 \\b[ABCD]\\b       （Daily-Omni）
      5) 若给了 index2ans → 全文匹配选项内容反查字母 （OmniBench）
    """
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()

    # 1) JSON
    m = re.search(r'"answer"\s*:\s*"([A-D])"', s)
    if m: return m.group(1)

    # 2) \boxed{}
    m = re.search(r'\\boxed\{(?:\\text\{)?([A-D])\}?\}', s)
    if m: return m.group(1)

    # 3) 首字符
    if s[0] in LETTERS:
        return s[0]

    # 4) 独立字母
    m = re.search(r'\b([A-D])\b', s)
    if m: return m.group(1)

    # 5) 选项内容反查
    if index2ans:
        s_low = s.lower()
        hits = [(letter, ans) for letter, ans in index2ans.items()
                if ans and ans.lower() in s_low]
        if len(hits) == 1:
            return hits[0][0]

    return None
```

---

## 3. 重试装饰器

```python
# unify_omnibench/utils/retry.py
import time, random, functools

def retry(max_retries=4, base_delay=4.0, jitter=1.0,
          retry_on=(Exception,), fatal_on=()):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except fatal_on as e:
                    raise
                except retry_on as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, jitter)
                    time.sleep(delay)
        return wrap
    return deco
```

`OpenAIChatModel.generate` 用 `@retry(...)` 包裹底层 `client.chat.completions.create`。

---

## 4. JSONL 原子追加（Resume 基石）

```python
# unify_omnibench/utils/io.py
import json, os, threading

_lock = threading.Lock()

def append_jsonl(path: str, record: dict):
    with _lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except json.JSONDecodeError: pass
    return out

def atomic_write_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
```

---

## 5. Runner 完整骨架

```python
# unify_omnibench/runner.py
import os, time, threading, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from tqdm import tqdm

from .core.types import InferenceRequest, InferenceResult, Sample
from .eval.parser import extract_choice_letter
from .eval.report import write_summary
from .concurrency.progress import ProgressManager
from .utils.io import append_jsonl, load_jsonl, atomic_write_json

class Runner:
    def __init__(self, dataset, model, cfg):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg
        self.run_dir = cfg["run_dir"]
        os.makedirs(self.run_dir, exist_ok=True)
        self.items_path  = os.path.join(self.run_dir, "items.jsonl")
        self.failed_path = os.path.join(self.run_dir, "failed.jsonl")

    # ---- public ----
    def run(self):
        self.model.load()
        atomic_write_json(os.path.join(self.run_dir, "run_config.yaml"), self.cfg)

        done_uids = self._load_done_uids()
        pending: List[Sample] = [s for s in self.dataset if s.uid not in done_uids]

        mode = self.cfg["concurrency"]["mode"]
        if   mode == "thread":     self._run_threaded(pending)
        elif mode == "batch":      self._run_batched(pending)
        elif mode == "sequential": self._run_sequential(pending)
        else: raise ValueError(f"Unknown concurrency mode: {mode}")

        summary = write_summary(self.items_path,
                                out_dir=self.run_dir,
                                dataset_name=self.cfg["dataset"]["name"])
        self.model.close()
        return summary

    # ---- internals ----
    def _load_done_uids(self) -> set:
        done = set()
        for rec in load_jsonl(self.items_path):
            if not rec.get("error"):           # 失败项不算 done，会被重新评测
                done.add(rec["uid"])
        return done

    def _build_req(self, sample: Sample) -> InferenceRequest:
        return InferenceRequest(
            sample=sample,
            modality_mode=self.cfg.get("modality_mode", "av"),
            prompt_template=self.cfg.get("prompt_template"),
            generation_kwargs=self.cfg.get("generation", {}),
        )

    def _infer_one(self, sample: Sample) -> InferenceResult:
        req = self._build_req(sample)
        t0 = time.time()
        try:
            raw = self.model.generate(req)
            parsed = extract_choice_letter(raw)
            return InferenceResult(
                uid=sample.uid, dataset=sample.dataset,
                raw_output=raw, parsed_answer=parsed,
                correct_answer=sample.answer,
                is_correct=(parsed is not None and parsed == sample.answer),
                latency_s=time.time() - t0,
                meta=sample.meta,
            )
        except Exception as e:
            return InferenceResult(
                uid=sample.uid, dataset=sample.dataset,
                raw_output="", error=f"{type(e).__name__}: {e}",
                latency_s=time.time() - t0,
                meta=sample.meta,
            )

    def _persist(self, res: InferenceResult):
        rec = res.__dict__.copy()
        append_jsonl(self.items_path, rec)
        if res.error:
            append_jsonl(self.failed_path, rec)

    # ---- modes ----
    def _run_sequential(self, pending: List[Sample]):
        for s in tqdm(pending, desc="Sequential"):
            self._persist(self._infer_one(s))

    def _run_threaded(self, pending: List[Sample]):
        W = self.cfg["concurrency"]["max_workers"]
        with ProgressManager(len(pending)) as pm, \
             ThreadPoolExecutor(max_workers=W) as pool:
            futures = [pool.submit(self._infer_one, s) for s in pending]
            for fut in as_completed(futures):
                res = fut.result()
                self._persist(res)
                pm.update(is_failed=bool(res.error),
                          is_correct=res.is_correct)

    def _run_batched(self, pending: List[Sample]):
        B = self.cfg["concurrency"]["batch_size"]
        for i in tqdm(range(0, len(pending), B), desc=f"Batch x{B}"):
            batch = pending[i:i + B]
            reqs = [self._build_req(s) for s in batch]
            try:
                raws = self.model.generate_batch(reqs)
            except Exception as e:
                # 整批失败 → 退化到顺序，避免整批被丢
                for s in batch:
                    self._persist(self._infer_one(s))
                continue
            for s, raw in zip(batch, raws):
                parsed = extract_choice_letter(raw)
                res = InferenceResult(
                    uid=s.uid, dataset=s.dataset, raw_output=raw,
                    parsed_answer=parsed, correct_answer=s.answer,
                    is_correct=(parsed is not None and parsed == s.answer),
                    meta=s.meta,
                )
                self._persist(res)
```

---

## 6. ProgressManager（端口自 OmniVideoBench）

```python
# unify_omnibench/concurrency/progress.py
import threading
from tqdm import tqdm

class ProgressManager:
    def __init__(self, total):
        self.total = total
        self.lock = threading.Lock()
        self.completed = 0
        self.correct = 0
        self.failed = 0
        self.bar = tqdm(total=total, desc="Eval",
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| '
                                   '{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')

    def update(self, is_failed=False, is_correct=False):
        with self.lock:
            if is_failed: self.failed += 1
            else:
                self.completed += 1
                if is_correct: self.correct += 1
            acc = self.correct / self.completed if self.completed else 0.0
            self.bar.set_postfix_str(
                f"Acc:{acc:.1%}({self.correct}/{self.completed}) Failed:{self.failed}")
            self.bar.update(1)

    def close(self): self.bar.close()
    def __enter__(self): return self
    def __exit__(self, *a): self.close()
```

---

## 7. Adapter 三例 (骨架)

### 7.1 Daily-Omni

```python
# unify_omnibench/datasets/daily_omni.py
import os, json
from typing import Iterator
from .base import BaseDatasetAdapter
from ..core.registry import register_dataset
from ..core.types import Sample, MediaRef

@register_dataset("daily_omni")
class DailyOmniAdapter(BaseDatasetAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        with open(cfg["qa_file"], "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.video_base = cfg["video_base_dir"]

    def __len__(self): return len(self.data)

    def __iter__(self) -> Iterator[Sample]:
        for idx, item in enumerate(self.data):
            vid = str(item["video_id"])
            video_path = os.path.join(self.video_base, vid, f"{vid}_video.mp4")
            audio_path = os.path.join(self.video_base, vid, f"{vid}_audio.wav")
            yield Sample(
                uid=self.make_uid(idx, vid),
                dataset=self.name,
                question=item["Question"],
                choices=item["Choice"],
                answer=item.get("Answer"),
                media=[MediaRef("video", video_path, mime="video/mp4"),
                       MediaRef("audio", audio_path, mime="audio/wav")],
                meta={
                    "video_id": vid,
                    "task_type": item.get("Type"),
                    "video_category": item.get("video_category"),
                    "video_duration": item.get("video_duration"),
                },
            )
```

### 7.2 OmniBench

```python
# unify_omnibench/datasets/omnibench.py
import os, json
from .base import BaseDatasetAdapter
from ..core.registry import register_dataset
from ..core.types import Sample, MediaRef

@register_dataset("omnibench")
class OmniBenchAdapter(BaseDatasetAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.records = self._load(cfg["data_file"])
        self.mm_root = cfg["mm_root"]

    def _load(self, path):
        if path.endswith(".jsonl"):
            with open(path) as f:
                return [json.loads(l) for l in f if l.strip()]
        elif path.endswith(".xlsx"):
            import pandas as pd
            return pd.read_excel(path).to_dict("records")
        else:
            raise ValueError(f"Unsupported file: {path}")

    def __len__(self): return len(self.records)

    def __iter__(self):
        for idx, r in enumerate(self.records):
            img_path = os.path.join(self.mm_root, "image", r["image_path"])
            aud_path = os.path.join(self.mm_root, "audio", r["audio_path"])
            yield Sample(
                uid=self.make_uid(r.get("index", idx)),
                dataset=self.name,
                question=r["question"],
                choices=r["options"] if isinstance(r["options"], list)
                                    else self._parse_options(r["option"]),
                answer=r.get("answer") or r.get("correct answer"),
                media=[MediaRef("image", img_path, mime="image/jpeg"),
                       MediaRef("audio", aud_path, mime="audio/wav")],
                meta={"task_type": r.get("task type"),
                      "audio_type": r.get("audio type")},
            )

    @staticmethod
    def _parse_options(s: str):
        import re
        pat = r'(A\..*?)(B\..*?)(C\..*?)(D\..*)'
        m = re.findall(pat, s, re.DOTALL)
        return [x.strip() for x in m[0]] if m else [s]
```

### 7.3 OmniVideoBench

```python
# unify_omnibench/datasets/omnivideobench.py
import os, json
from .base import BaseDatasetAdapter
from ..core.registry import register_dataset
from ..core.types import Sample, MediaRef

def _mmss_to_sec(t: str) -> int:
    parts = t.split(":")
    if len(parts) == 2: return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(t)

@register_dataset("omnivideobench")
class OmniVideoBenchAdapter(BaseDatasetAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        with open(cfg["data_file"]) as f:
            self.data = json.load(f)
        self.video_dir = cfg["video_dir"]
        # 预展开
        self._samples = list(self._expand())

    def _expand(self):
        for v_idx, v in enumerate(self.data):
            vname = v["video"]
            vpath = os.path.join(self.video_dir, f"{vname}.mp4")
            dur = _mmss_to_sec(v["duration"]) if v.get("duration") else None
            for q_idx, q in enumerate(v.get("questions", [])):
                yield Sample(
                    uid=self.make_uid(v_idx, vname, q_idx),
                    dataset=self.name,
                    question=q["question"],
                    choices=q["options"],
                    answer=q.get("correct_option"),
                    media=[MediaRef("video", vpath, mime="video/mp4",
                                    extra={"duration_s": dur})],
                    meta={
                        "video": vname,
                        "video_type": v.get("video_type"),
                        "question_type": q.get("question_type"),
                        "audio_type": q.get("audio_type"),
                        "duration_s": dur,
                    },
                )

    def __iter__(self): return iter(self._samples)
    def __len__(self):  return len(self._samples)
```

---

## 8. Model 三例 (骨架)

### 8.1 OpenAI 兼容

```python
# unify_omnibench/models/api/openai_chat.py
import os, base64
from typing import List
from openai import OpenAI
from ..base import BaseModel
from ...core.registry import register_model
from ...core.types import InferenceRequest, MediaRef
from ...media.video_io import extract_frames_base64
from ...utils.retry import retry

@register_model("openai_chat")
class OpenAIChatModel(BaseModel):
    supports_modalities = ("video", "image", "audio", "text")
    is_thread_safe = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.api_key = os.environ.get(cfg["api_key_env"], cfg.get("api_key", ""))
        self.base_url = os.environ.get(cfg["base_url_env"], cfg.get("base_url", ""))
        self.model_name = cfg["model"]
        self.client: OpenAI | None = None
        self.seconds_per_frame = cfg.get("video", {}).get("seconds_per_frame", 2)
        self.max_frames = cfg.get("video", {}).get("max_frames", 32)
        self.retry_kwargs = cfg.get("retry", {"max_retries": 4, "base_delay": 4})

    def load(self):
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _media_to_content(self, media: List[MediaRef], modality_mode: str):
        out = []
        for m in media:
            if modality_mode == "visual" and m.kind == "audio": continue
            if modality_mode == "audio"  and m.kind == "video": continue
            if modality_mode == "text": continue

            if m.kind == "image":
                with open(m.path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                out.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            elif m.kind == "video":
                frames = extract_frames_base64(m.path, self.seconds_per_frame,
                                               max_frames=self.max_frames)
                for fb in frames:
                    out.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{fb}"}})
            elif m.kind == "audio":
                # 视模型能力：直接 input_audio 或预先转写
                with open(m.path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                out.append({"type": "input_audio",
                            "input_audio": {"data": b64, "format": "wav"}})
        return out

    def build_messages(self, req: InferenceRequest):
        s = req.sample
        choices_text = "\n".join(s.choices) if isinstance(s.choices, list) else str(s.choices)
        prompt = (
            "Answer the multiple-choice question. Respond with a single capital letter (A/B/C/D).\n\n"
            f"Question: {s.question}\nOptions:\n{choices_text}"
        )
        content = self._media_to_content(s.media, req.modality_mode)
        content.append({"type": "text", "text": prompt})
        return [
            {"role": "system",
             "content": "You are a multimodal evaluator. Answer with one letter only."},
            {"role": "user", "content": content},
        ]

    def generate(self, req: InferenceRequest) -> str:
        messages = self.build_messages(req)

        @retry(**self.retry_kwargs)
        def _call():
            comp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=req.generation_kwargs.get("temperature", 0.0),
                max_tokens=req.generation_kwargs.get("max_new_tokens", 16),
            )
            return comp.choices[0].message.content or ""

        return _call().strip()
```

### 8.2 Gemini

```python
# unify_omnibench/models/api/gemini.py
import os, threading, time
from google import genai
from google.genai import types
from ..base import BaseModel
from ...core.registry import register_model
from ...core.types import InferenceRequest
from ...utils.retry import retry

@register_model("gemini")
class GeminiModel(BaseModel):
    supports_modalities = ("video", "image", "audio", "text")
    is_thread_safe = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.api_key = os.environ.get(cfg["api_key_env"], cfg.get("api_key", ""))
        self.model_name = cfg["model"]
        self._local = threading.local()

    def load(self):
        # 真正 client 在每个线程懒加载，保证 thread-local
        pass

    def _client(self):
        if not hasattr(self._local, "c"):
            self._local.c = genai.Client(api_key=self.api_key)
        return self._local.c

    def _upload(self, path: str):
        c = self._client()
        f = c.files.upload(file=path)
        retry = 0
        while f.state.name == "PROCESSING" and retry < 100:
            time.sleep(5)
            f = c.files.get(name=f.name); retry += 1
        if f.state.name == "FAILED":
            raise RuntimeError(f"Upload failed: {path}")
        return f

    def generate(self, req: InferenceRequest) -> str:
        s = req.sample
        # 仅传第一个 video/image（Gemini 直接消化媒体）；按 modality_mode 过滤
        media = [m for m in s.media
                 if not (req.modality_mode == "visual" and m.kind == "audio")
                 and not (req.modality_mode == "audio"  and m.kind == "video")]

        uploaded = [self._upload(m.path) for m in media] if req.modality_mode != "text" else []
        prompt = (
            "Answer the multiple-choice question. Reply with one letter A/B/C/D.\n"
            f"Question: {s.question}\nOptions:\n" +
            "\n".join(s.choices if isinstance(s.choices, list) else [str(s.choices)])
        )

        @retry(max_retries=4, base_delay=4)
        def _call():
            r = self._client().models.generate_content(
                model=self.model_name,
                contents=[*uploaded, prompt],
                config=types.GenerateContentConfig(
                    temperature=req.generation_kwargs.get("temperature", 0.0)),
            )
            return r.text or ""

        try:
            return _call().strip()
        finally:
            for f in uploaded:
                try: self._client().files.delete(name=f.name)
                except Exception: pass
```

### 8.3 本地 Transformers (Qwen2.5-Omni 示例)

```python
# unify_omnibench/models/local/qwen25omni.py
import torch
from ..base import BaseModel
from ...core.registry import register_model
from ...core.types import InferenceRequest

@register_model("transformers_qwen25omni")
class Qwen25OmniModel(BaseModel):
    supports_modalities = ("video", "audio", "image", "text")
    is_thread_safe = False
    supports_batch = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_path = cfg["model_name_or_path"]
        self.device_map = cfg.get("device", "auto")
        self.attn_impl  = cfg.get("attn_implementation", "flash_attention_2")
        self.max_frames = cfg.get("max_frames", 256)

    def load(self):
        from transformers import (Qwen2_5OmniForConditionalGeneration,
                                  Qwen2_5OmniProcessor)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16,
            device_map=self.device_map, attn_implementation=self.attn_impl)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)

    def _build_conv(self, req: InferenceRequest):
        s = req.sample
        opts = "\n".join(s.choices) if isinstance(s.choices, list) else str(s.choices)
        text = (f"Question:\n{s.question}\n\nOptions:\n{opts}\n\n"
                "Answer with the option's letter directly (A/B/C/D).")
        content = []
        for m in s.media:
            if req.modality_mode == "visual" and m.kind == "audio": continue
            if req.modality_mode == "audio"  and m.kind == "video": continue
            if   m.kind == "video": content.append({"type": "video", "video": m.path})
            elif m.kind == "image": content.append({"type": "image", "image": m.path})
            elif m.kind == "audio": content.append({"type": "audio", "audio": m.path})
        content.append({"type": "text", "text": text})
        return [
            {"role": "system",
             "content": [{"type": "text", "text": "You are Qwen2.5-Omni."}]},
            {"role": "user", "content": content},
        ]

    def generate(self, req: InferenceRequest) -> str:
        # 这里直接复用 Daily-Omni 已有的 process_mm_info、抽帧采样逻辑
        from ...media.qwen_utils import process_mm_info, sample_frames
        conversation = self._build_conv(req)
        text = self.processor.apply_chat_template(conversation,
                                                  add_generation_prompt=True, tokenize=False)
        use_audio_in_video = (req.modality_mode in ("av", "audio"))
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=use_audio_in_video)
        videos = sample_frames(videos, self.max_frames)

        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True,
                                use_audio_in_video=use_audio_in_video)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with torch.no_grad():
            out_ids, _ = self.model.generate(
                **inputs, use_audio_in_video=use_audio_in_video,
                max_new_tokens=req.generation_kwargs.get("max_new_tokens", 10),
                do_sample=req.generation_kwargs.get("do_sample", False),
                temperature=req.generation_kwargs.get("temperature", 0.0))
        return self.processor.batch_decode(out_ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
```

### 8.4 vLLM (统一批处理)

```python
# unify_omnibench/models/vllm/vllm_runner.py
from typing import List
from ..base import BaseModel
from ...core.registry import register_model
from ...core.types import InferenceRequest

@register_model("vllm")
class VLLMModel(BaseModel):
    is_thread_safe = False
    supports_batch = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg_model = cfg

    def load(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=self.cfg_model["model"],
            tensor_parallel_size=self.cfg_model.get("tensor_parallel_size", 1),
            gpu_memory_utilization=self.cfg_model.get("gpu_memory_utilization", 0.9),
            limit_mm_per_prompt=self.cfg_model.get("limit_mm_per_prompt"),
            max_num_seqs=self.cfg_model.get("max_num_seqs", 8),
        )
        self.SamplingParams = SamplingParams

    def _build_prompt(self, req: InferenceRequest):
        # 按模型族构造（与 transformers 模型一致的 chat_template）
        ...

    def generate(self, req: InferenceRequest) -> str:
        return self.generate_batch([req])[0]

    def generate_batch(self, reqs: List[InferenceRequest]) -> List[str]:
        prompts = [self._build_prompt(r) for r in reqs]
        sp = self.SamplingParams(
            max_tokens=reqs[0].generation_kwargs.get("max_new_tokens", 16),
            temperature=reqs[0].generation_kwargs.get("temperature", 0.0))
        outs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs]
```

> 注：vLLM 多模态的 prompt 构造与 `transformers` 略有不同，建议每个模型族（Qwen-Omni / Qwen-VL / VideoLLaMA2）做一个 `VLLMQwenOmniModel` 子类，统一 `_build_prompt`。

---

## 9. 评估与报告

```python
# unify_omnibench/eval/report.py
import os, json
from collections import defaultdict
from ..utils.io import load_jsonl, atomic_write_json

def write_summary(items_path: str, out_dir: str, dataset_name: str):
    items = load_jsonl(items_path)
    total = len(items)
    failed = sum(1 for x in items if x.get("error"))
    valid = [x for x in items if not x.get("error") and x.get("parsed_answer")]
    correct = sum(1 for x in valid if x.get("is_correct"))

    by = defaultdict(lambda: {"n": 0, "c": 0})
    for x in valid:
        for key in ("task_type", "question_type", "audio_type",
                    "video_category", "duration_s"):
            v = (x.get("meta") or {}).get(key)
            if v is None: continue
            bk = f"{key}={v}"
            by[bk]["n"] += 1
            by[bk]["c"] += int(x["is_correct"])

    summary = {
        "dataset": dataset_name,
        "total": total,
        "failed": failed,
        "valid": len(valid),
        "correct": correct,
        "accuracy": correct / len(valid) if valid else 0.0,
        "breakdown": {k: {"n": v["n"], "c": v["c"],
                          "acc": v["c"] / v["n"] if v["n"] else 0.0}
                      for k, v in by.items()},
    }
    atomic_write_json(os.path.join(out_dir, "summary.json"), summary)

    # markdown
    lines = [f"# Summary: {dataset_name}",
             f"- total={total} valid={len(valid)} failed={failed}",
             f"- accuracy={summary['accuracy']:.2%} ({correct}/{len(valid)})",
             "", "## Breakdown", "| key | acc | correct/total |", "|---|---:|---:|"]
    for k, v in sorted(summary["breakdown"].items()):
        lines.append(f"| {k} | {v['acc']:.2%} | {v['c']}/{v['n']} |")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    return summary
```

---

## 10. CLI 入口

```python
# unify_omnibench/cli.py
import argparse, os, time, yaml
from .core.registry import build_dataset, build_model
from .runner import Runner

def _load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("_base_", None)
    if base:
        with open(base) as f:
            merged = yaml.safe_load(f)
        _deep_update(merged, cfg)
        return merged
    return cfg

def _deep_update(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else: a[k] = v

def cmd_run(args):
    cfg = _load_cfg(args.config)
    run_name = cfg.get("run_name", "run")
    cfg["run_dir"] = os.path.join(cfg.get("output_dir", "runs"),
                                  f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    ds = build_dataset(cfg["dataset"])
    md = build_model(cfg["model"])
    Runner(ds, md, cfg).run()

def cmd_rerun_failed(args):
    # 读取 failed.jsonl 的 uid 集合 → 把 items.jsonl 中这些 uid 行删除（或标记）→ run()
    ...

def cmd_report(args):
    ...

def main():
    p = argparse.ArgumentParser(prog="unify-eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run");           pr.add_argument("--config", required=True); pr.set_defaults(fn=cmd_run)
    pf = sub.add_parser("rerun-failed");  pf.add_argument("--run_dir", required=True); pf.set_defaults(fn=cmd_rerun_failed)
    pg = sub.add_parser("report");        pg.add_argument("--runs", nargs="+", required=True); pg.set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
```

`pyproject.toml`：

```toml
[project.scripts]
unify-eval = "unify_omnibench.cli:main"
```

---

## 11. 与 Daily-Omni 已有代码的复用建议

避免重写已经能跑的本地推理逻辑，建议直接 `import`：

| 想复用 | 推荐做法 |
|---|---|
| `Daily-Omni/test_model/Qwen2.5-Omni/testmodel.py` 里的 `process_mm_info` / 抽帧 / vLLM 路径 | 抽到 `unify_omnibench/media/qwen_utils.py`，在 `Qwen25OmniModel.generate` 中直接调用 |
| `Daily-Omni/test_model_api/test_utils.py` 的 ffmpeg / base64 / extract_frames_base64 | 抽到 `unify_omnibench/media/video_io.py`（去掉 `config` 耦合，改为参数传入） |
| `OmniVideoBench/eval/gemini_eval.py` 的 `ThreadLocalGeminiClient` / `ProgressManager` | 已端口到 `models/api/gemini.py` 和 `concurrency/progress.py` |
| `OmniBench/inference/answer_parsing.py` 的 `parse_multi_choice_response` | 已融合进 `eval/parser.py` 的 step 5 |

---

## 12. 测试用例建议

- `tests/test_parser.py`：覆盖 5 类输出 → 正确字母；
- `tests/test_adapters.py`：每个 Adapter 用 5 条 fixture，断言 uid/answer/media 路径合理；
- `tests/test_runner_resume.py`：mock Model（每隔 3 条抛异常），跑两次，断言第二次只跑 failed；
- `tests/test_concurrency.py`：mock Model 计时，验证 ThreadPool 比 sequential 快 ≥ 3×；
- 集成测试：在 `example_videos/` (Daily-Omni 自带的 12 个小视频) 上跑 `transformers_qwen25omni`，看 accuracy 不报错。

---

## 13. 给实现者的下一步 Checklist

- [ ] 初始化 `pyproject.toml` + 包目录结构
- [ ] 实现 `core/types.py` + `registry.py` + `config.py`
- [ ] 实现 `eval/parser.py` 并跑通单测
- [ ] 实现 `DailyOmniAdapter` + dummy `EchoModel`（直接返回 "A"）→ 跑通端到端
- [ ] 实现 `OpenAIChatModel` + `_run_threaded` → 跑 GPT-4o on Daily-Omni 30 条验证
- [ ] 实现 `Qwen25OmniModel`（复用 Daily-Omni 现成代码）→ 在 `example_videos/` 上跑
- [ ] 加入 `OmniBenchAdapter` / `OmniVideoBenchAdapter`
- [ ] 加入 `VLLMModel` + `_run_batched`
- [ ] 编写 `summary.md` 模板 + `unify-eval report` 多 run 横向对比

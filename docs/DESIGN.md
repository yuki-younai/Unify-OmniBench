# Unify-OmniBench 设计方案

> 一个统一的 Omni-modal Benchmark 评测框架：把 **Daily-Omni / OmniBench / OmniVideoBench** 等异构的多模态评测集，通过统一的数据接口、模型接口和运行流程整合到同一个 CLI 入口下，支持本地 (Transformers / vLLM) 与外部 API (OpenAI 兼容、Gemini 等) 两类模型，并对 API 提供并发处理与断点续跑能力。

---

## 1. 设计目标

| 维度 | 目标 |
|---|---|
| **统一数据加载** | 任意 Benchmark 通过一个 `Adapter` 把原生数据格式映射为统一的 `Sample` 对象（含 video / audio / image / text，以及 question / options / answer / meta） |
| **统一模型接口** | 任意模型实现统一的 `BaseModel.generate(sample) -> str`；支持 OpenAI 兼容 API、本地 `transformers`、本地 `vllm` 三种后端 |
| **统一运行流程** | 一条命令完成：加载数据 → 构造请求 → 推理 → 抽取答案 → 评估 → 持久化结果；支持 resume |
| **统一评估指标** | 多选题精确匹配（A/B/C/D），按 task_type / modality / duration / dataset 等维度分桶统计；可扩展自由问答 (LLM-as-Judge) |
| **并发执行** | 外部 API 走 **线程池**（IO 密集）；本地 transformers 走单进程；本地 vLLM 走 batch；CPU 预处理可走进程池 |
| **可复现** | 一切配置 YAML 化；每次 run 落盘 `run_config.yaml` + `*_items.jsonl` + `summary.json` |

---

## 2. 三个现有 Benchmark 的共性与差异（共性即抽象点）

| 项目 | 数据格式 | 模态 | 输入媒体来源 | 评测方式 | 已有并发 |
|---|---|---|---|---|---|
| **Daily-Omni** | `qa.json`（list of dict, 包含 `Question/Choice/Answer/video_id/Type/video_category/video_duration`） | video + audio (mp4 + wav)、可选 visual-only / audio-only | `video_id` 拼接 `{video_id}/{video_id}_video.mp4` | 抽 A/B/C/D 与 `Answer` 对比 | API 端 `ProcessPoolExecutor` |
| **OmniBench** | `batch-*.jsonl/xlsx`（`question/option/correct answer/audio_path/image_path/audio type/task type`） | image + audio + text | `mm_data/image/...`、`mm_data/audio/...` | `parse_multi_choice_response` 抽 A/B/C/D | API 端 `multiprocessing.Pool` |
| **OmniVideoBench** | `data.json`（嵌套：video → questions[]，含 `options/correct_option/question_type/audio_type/duration`） | video (含音轨) | `video_dir/{video}.mp4` | clean_text 精确匹配选项字母 | `ThreadPoolExecutor` + `ProgressManager` + resume |

**共性抽象**：

- **样本 = (id, media[video/audio/image], question, choices, answer, meta)**
- **媒体可选**（modality_mode 控制：`av` / `visual` / `audio` / `text`）
- **输出 = 单字母 A/B/C/D**（选项字母评估即可覆盖三者主线）
- **耗时主因 = API IO 或 本地 GPU**（决定并发策略不同）
- **必须支持 resume**（OmniVideoBench 和 OmniBench 都自实现了一遍 → 抽到框架层）

---

## 3. 顶层架构

```
                ┌─────────────────────────────────────────────────────┐
                │                  Unify-OmniBench CLI                │
                │              unify-eval run --config x.yaml         │
                └──────────────────────────┬──────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
        ┌──────────┐               ┌───────────────┐            ┌──────────────┐
        │ Dataset  │               │   Runner      │            │   Reporter   │
        │ Adapter  │──Sample────►──│  (Engine)     │──Result───►│ (metrics +   │
        │          │               │               │            │  per-item)   │
        └──────────┘               └───────┬───────┘            └──────────────┘
        DailyOmniAdapter                   │
        OmniBenchAdapter                   │ generate(prompt, media)
        OmniVideoBenchAdapter              ▼
        (HF / JSON / JSONL)        ┌───────────────┐
                                   │  Model        │
                                   │  (统一接口)    │
                                   └───────┬───────┘
                                           │
                       ┌───────────────────┼────────────────────┐
                       ▼                   ▼                    ▼
              ┌─────────────────┐ ┌──────────────────┐ ┌──────────────────┐
              │ OpenAIChatModel │ │ TransformersModel│ │   VLLMModel      │
              │ (兼容GPT/Qwen/  │ │ (Qwen-Omni,VL,   │ │ (vllm.LLM,本地   │
              │  DeepSeek/...)  │ │  VideoLLaMA2,...)│ │  批量高吞吐)      │
              │ GeminiModel     │ │                  │ │                  │
              └─────────────────┘ └──────────────────┘ └──────────────────┘
                  线程池并发           单进程逐条             批处理
```

四层职责清晰：

- **Adapter**：只关心“怎么把 X benchmark 的原始文件 → `Sample` 流”。
- **Model**：只关心“给我一个 `Sample`、一个 `Prompt`，我返回模型字符串输出”。
- **Runner**：编排迭代、并发、重试、resume、保存。
- **Reporter**：评测、分桶统计、写报告。

---

## 4. 核心数据结构

```python
# unify_omnibench/core/types.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal

Modality = Literal["video", "audio", "image", "text"]

@dataclass
class MediaRef:
    """媒体资源的轻量描述，运行时按需加载/编码"""
    kind: Modality
    path: str                       # 本地路径
    mime: Optional[str] = None      # 例 "video/mp4" / "audio/wav" / "image/jpeg"
    extra: Dict[str, Any] = field(default_factory=dict)  # duration, fps, sample_rate ...

@dataclass
class Sample:
    """统一样本对象 —— 所有 Adapter 都产出它"""
    uid: str                                # 全局唯一 id（{dataset}:{index}:{video_id}）
    dataset: str                            # daily_omni / omnibench / omnivideobench
    question: str
    choices: List[str]                      # ["A. xxx", "B. xxx", ...] 或 ["xxx", "xxx", ...]
    answer: Optional[str] = None            # 标准答案字母 "A"/"B"/"C"/"D"
    media: List[MediaRef] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)  # task_type, duration, audio_type ...

@dataclass
class InferenceRequest:
    """Runner 提交给 Model 的请求"""
    sample: Sample
    modality_mode: Literal["av", "visual", "audio", "text"] = "av"
    prompt_template: Optional[str] = None
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)  # max_new_tokens, temperature ...

@dataclass
class InferenceResult:
    uid: str
    dataset: str
    raw_output: str                          # 模型原始字符串
    parsed_answer: Optional[str] = None      # 抽出的字母
    correct_answer: Optional[str] = None
    is_correct: bool = False
    error: Optional[str] = None              # 失败原因
    latency_s: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
```

---

## 5. 模块设计

### 5.1 Dataset Adapter（统一数据加载）

```python
# unify_omnibench/datasets/base.py
class BaseDatasetAdapter:
    name: str

    def __init__(self, cfg: Dict[str, Any]): ...

    def __iter__(self) -> Iterator[Sample]: ...

    def __len__(self) -> int: ...
```

为三个 Benchmark 各写一个 Adapter：

- `DailyOmniAdapter`：读 `qa.json` → 每条产生一个 `Sample`，`media = [video, audio]`（路径按 `BASE_VIDEO_DIR/{video_id}/{video_id}_video.mp4`、`..._audio.wav` 规则拼接）。
- `OmniBenchAdapter`：读 `batch-*.jsonl` 或 `load_dataset("m-a-p/OmniBench")` → `media = [image, audio]`。
- `OmniVideoBenchAdapter`：读嵌套 JSON，沿用 `VideoQADaloader.get_all_qa_pairs` 的展开逻辑 → `media = [video]`，meta 含 `question_type / audio_type / duration_seconds`。

**注册机制**（避免 if/elif 列表）：

```python
DATASET_REGISTRY = {}
def register_dataset(name):
    def deco(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return deco
```

CLI 通过 `dataset.name` 反查类。

---

### 5.2 Model 接口（统一模型接口）

```python
# unify_omnibench/models/base.py
class BaseModel:
    name: str
    supports_modalities: Tuple[Modality, ...]   # 该模型支持的输入媒体
    is_thread_safe: bool = False                # 决定 Runner 是否允许多线程

    def __init__(self, cfg: Dict[str, Any]): ...

    def load(self): ...                          # 懒加载/初始化
    def close(self): ...                         # 释放显存等

    def build_messages(self, req: InferenceRequest) -> Any:
        """把 Sample → 各后端要的输入（OpenAI messages / HF conversation / vLLM prompt）"""

    def generate(self, req: InferenceRequest) -> str:
        """同步返回模型 raw 文本输出"""

    # 可选：批量接口（vLLM 默认实现，其它后端默认 fallback 到 for 循环）
    def generate_batch(self, reqs: List[InferenceRequest]) -> List[str]:
        return [self.generate(r) for r in reqs]
```

**三类内置后端**：

#### (a) `OpenAIChatModel`（OpenAI 兼容 API：GPT-4o / DeepSeek / Qwen-Bailian / 本地 vLLM `--served-model-name` 暴露的 OpenAI server）

- 用 `openai.OpenAI(api_key, base_url)`；
- 媒体编码策略：
  - `image` → base64 `data:image/jpeg;base64,...`
  - `video` → 抽帧 (`ffmpeg`/`cv2`，参数 `seconds_per_frame`) → 多张 image_url
  - `audio` → 走 OpenAI `input_audio` 或转写后注入（按模型能力开关）
- 重试：`MAX_RETRIES + 指数退避 + jitter`（参考 `test_utils._call_openai_compatible_api`）。
- `is_thread_safe = True`，由 Runner 用 `ThreadPoolExecutor` 并发。

#### (b) `GeminiModel`（google-genai）

- 借鉴 `OmniVideoBench/eval/gemini_eval.py` 的 `ThreadLocalGeminiClient`，每线程一个 client；
- 直传 video 文件 `files.upload`，无须抽帧；`no_sound` 走 ffmpeg 去音轨。
- `is_thread_safe = True`。

#### (c) `TransformersModel`（本地 `transformers` 推理）

- 一个抽象基类 + 每个模型族一个子类（`Qwen25OmniModel` / `Qwen3OmniModel` / `Qwen25VLModel` / `VideoLLaMA2Model` / ...），复用 Daily-Omni `test_model/*/testmodel.py` 中现成的 conversation 构造与 `process_mm_info` 逻辑；
- `is_thread_safe = False`（单 GPU 进程），Runner 走顺序 + 进度条。

#### (d) `VLLMModel`（本地 vLLM）

- 用 `vllm.LLM(model=..., gpu_memory_utilization=..., limit_mm_per_prompt={"image":N,"video":1,"audio":1})`；
- 借鉴 Daily-Omni `Qwen2.5-Omni` 已有的 `--use_vllm` 路径；
- 通过 `generate_batch` 做大批量推理（Runner 自动批处理）；
- `is_thread_safe = False`，但天然支持 batch。

**注册机制** 同 Adapter：`MODEL_REGISTRY[name] = cls`。

---

### 5.3 Runner（统一运行流程）

```python
# unify_omnibench/runner.py
class Runner:
    def __init__(self, dataset, model, cfg):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg
        self.output_dir = cfg["output_dir"]
        self.result_path = os.path.join(self.output_dir, "items.jsonl")

    def run(self):
        # 1) resume：扫描 items.jsonl，记录已完成的 uid 集合
        done = self._load_done_uids()
        pending = [s for s in self.dataset if s.uid not in done]

        # 2) 选并发策略
        if self.model.is_thread_safe and self.cfg["concurrency"]["mode"] == "thread":
            self._run_threaded(pending)
        elif hasattr(self.model, "generate_batch") and self.cfg["concurrency"]["mode"] == "batch":
            self._run_batched(pending)
        else:
            self._run_sequential(pending)

        # 3) 汇总
        return self._aggregate()
```

#### 并发策略矩阵

| 模型类别 | concurrency.mode | 实现 |
|---|---|---|
| OpenAI / Gemini API | `thread` | `ThreadPoolExecutor(max_workers=W)` + 每完成 N 条/30s 增量落盘 jsonl |
| 本地 vLLM | `batch` | 攒一个 `batch_size` 再 `generate_batch` |
| 本地 transformers | `sequential` | 单线程逐条，tqdm |
| 视频预处理（抽帧/转码）重 | `process` (可选) | `ProcessPoolExecutor` 预处理 → 主进程做模型推理 |

并发执行的核心代码（参考 `gemini_eval._execute_true_multithreaded_evaluation`）：

```python
def _run_threaded(self, pending):
    W = self.cfg["concurrency"]["max_workers"]
    save_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=W) as pool, \
         ProgressManager(len(pending)) as pm:
        futures = {pool.submit(self._infer_one, s): s for s in pending}
        last_save = time.time()
        for fut in as_completed(futures):
            res = fut.result()
            self._append_jsonl(res, save_lock)
            pm.update(res)
            if time.time() - last_save >= self.cfg["save_interval_s"]:
                self._snapshot(save_lock); last_save = time.time()

def _infer_one(self, sample: Sample) -> InferenceResult:
    req = InferenceRequest(sample, modality_mode=self.cfg["modality_mode"],
                           generation_kwargs=self.cfg["generation"])
    t0 = time.time()
    try:
        raw = self.model.generate(req)
        parsed = extract_choice_letter(raw)
        return InferenceResult(uid=sample.uid, dataset=sample.dataset,
                               raw_output=raw, parsed_answer=parsed,
                               correct_answer=sample.answer,
                               is_correct=(parsed == sample.answer),
                               latency_s=time.time()-t0)
    except Exception as e:
        return InferenceResult(uid=sample.uid, dataset=sample.dataset,
                               raw_output="", error=repr(e),
                               latency_s=time.time()-t0)
```

#### 重试 / 退避

- 在 Model 层（`OpenAIChatModel`、`GeminiModel`）内部封装 `MAX_RETRIES + 指数退避`；
- Runner 层只做 **失败结果记录**与最终 **一键重跑失败项**（`unify-eval rerun-failed`）。

#### 断点续跑

- 结果文件 `items.jsonl` 每行：`{"uid": ..., "is_correct": ..., ...}`；
- 启动时读取 → 构建 `done = {uid for uid in jsonl if error is None}`；
- 失败项 (`error != None`) 自动重投，覆盖原 line（按 uid）。

---

### 5.4 Reporter（统一评估）

`unify_omnibench/eval/`：

- `parser.py`：`extract_choice_letter(text) -> "A"/"B"/"C"/"D"/None`，融合 Daily-Omni `extract_choice_letter` + OmniBench `parse_multi_choice_response`（先严格首字母 → \b A-D \b → \boxed{X}）。
- `metrics.py`：
  - `overall_accuracy`
  - `accuracy_by(field)`：按 `meta.task_type / meta.audio_type / meta.duration_bucket / meta.video_category` 分桶；
  - 失败率统计（API 失败 / 媒体缺失 / 解析失败）；
- `report.py`：输出 `summary.json` + `summary.md`（带表格）+ 控制台打印（沿用 `test_utils.print_statistics` 的风格）。

---

## 6. CLI 与配置

### 6.1 CLI

```bash
# 评测
unify-eval run --config configs/qwen25omni_dailyomni_av.yaml

# 仅重跑失败项
unify-eval rerun-failed --run_dir runs/2026xxxx-xxxxxx/

# 汇总多个 run 出对比表
unify-eval report --runs runs/* --out leaderboard.md
```

底层用 `argparse` + 子命令；也可同时支持纯 `python -m unify_omnibench.cli run --config ...`。

### 6.2 配置文件（YAML，单文件描述一次 run）

```yaml
# configs/qwen25omni_dailyomni_av.yaml
run_name: qwen25omni-3b_dailyomni_av
output_dir: runs/                  # 实际写入 runs/{run_name}_{timestamp}/

dataset:
  name: daily_omni                 # 注册名
  qa_file: /data/Daily-Omni/qa.json
  video_base_dir: /data/Daily-Omni/Videos
  # 可选：max_items / filter_duration / shuffle / seed

modality_mode: av                  # av | visual | audio | text

model:
  name: transformers_qwen25omni    # 注册名
  model_name_or_path: /models/Qwen2.5-Omni-3B
  device: auto
  attn_implementation: flash_attention_2
  use_audio_in_video: true
  max_frames: 256

generation:
  max_new_tokens: 10
  temperature: 0.0
  do_sample: false

concurrency:
  mode: sequential                 # sequential | thread | batch | process
  max_workers: 1
  batch_size: 1

save_interval_s: 30
log_level: INFO
```

```yaml
# configs/gpt4o_omnibench_imgaudio.yaml —— API 模型 + 并发
run_name: gpt4o_omnibench_imgaudio
output_dir: runs/

dataset:
  name: omnibench
  data_file: /data/OmniBench/dataset/batch-5_1142_20240817.jsonl
  mm_root: /data/OmniBench/mm_data

modality_mode: av

model:
  name: openai_chat
  api_key_env: GPT4O_API_KEY
  base_url_env: GPT4O_BASE_URL
  model: gpt-4o-2024-08-06
  video:
    seconds_per_frame: 2
    max_frames: 32
  audio:
    mode: transcribe_text          # transcribe_text | input_audio
  retry:
    max_retries: 4
    base_delay: 4
  request_timeout_s: 60

generation:
  max_new_tokens: 20
  temperature: 0.0

concurrency:
  mode: thread
  max_workers: 16

save_interval_s: 30
```

```yaml
# configs/vllm_qwen3omni_omnivideobench.yaml —— 本地 vLLM 批处理
model:
  name: vllm
  model: /models/Qwen3-Omni-30B-A3B-Instruct
  tensor_parallel_size: 4
  gpu_memory_utilization: 0.92
  limit_mm_per_prompt: {video: 1, audio: 1}
  max_num_seqs: 8

concurrency:
  mode: batch
  batch_size: 8
```

### 6.3 配置加载/合并

- 支持 `_base_: configs/_base/dailyomni.yaml` 类似的继承（参考 mmcv）；
- 命令行 `--override model.generation.temperature=0.2` 用 `OmegaConf` 或简单点 `argparse + dotted key` 覆盖。

---

## 7. 目录结构

```
Unify-OmniBench/
├── DESIGN.md                      # 本文档
├── ARCHITECTURE.md                # 接口与目录细则
├── ADAPTERS.md                    # 新增数据集 / 模型的指南
├── README.md                      # 用户向 quickstart
├── pyproject.toml                 # 包装为 pip install -e .
├── configs/
│   ├── _base/
│   │   ├── dailyomni.yaml
│   │   ├── omnibench.yaml
│   │   └── omnivideobench.yaml
│   ├── qwen25omni_dailyomni_av.yaml
│   ├── gpt4o_omnibench_imgaudio.yaml
│   └── vllm_qwen3omni_omnivideobench.yaml
├── unify_omnibench/
│   ├── __init__.py
│   ├── cli.py                     # entry: unify-eval run/rerun-failed/report
│   ├── core/
│   │   ├── types.py               # Sample / MediaRef / InferenceRequest / InferenceResult
│   │   ├── registry.py            # DATASET_REGISTRY / MODEL_REGISTRY
│   │   ├── config.py              # YAML 加载、_base_ 继承、CLI 覆盖
│   │   └── logging.py
│   ├── datasets/
│   │   ├── base.py                # BaseDatasetAdapter
│   │   ├── daily_omni.py
│   │   ├── omnibench.py
│   │   ├── omnivideobench.py
│   │   └── hf_loader.py           # 通用 HuggingFace datasets 加载工具
│   ├── models/
│   │   ├── base.py                # BaseModel
│   │   ├── api/
│   │   │   ├── openai_chat.py     # 兼容 GPT-4o/DeepSeek/Qwen API/本地 vLLM-OpenAI
│   │   │   └── gemini.py
│   │   ├── local/
│   │   │   ├── transformers_base.py
│   │   │   ├── qwen25omni.py
│   │   │   ├── qwen3omni.py
│   │   │   ├── qwen25vl.py
│   │   │   ├── qwen3vl.py
│   │   │   └── videollama2.py
│   │   └── vllm/
│   │       └── vllm_runner.py
│   ├── media/
│   │   ├── video_io.py            # 抽帧、去音轨、转码（封装 ffmpeg/cv2）
│   │   ├── audio_io.py
│   │   └── encode.py              # base64、tempfile 管理
│   ├── runner.py                  # Runner 实现，调度并发/resume
│   ├── concurrency/
│   │   ├── threaded.py
│   │   ├── batched.py
│   │   └── progress.py            # ProgressManager（端口 OmniVideoBench 的实现）
│   ├── eval/
│   │   ├── parser.py              # extract_choice_letter
│   │   ├── metrics.py
│   │   ├── report.py
│   │   └── llm_judge.py           # 可选：自由问答评估
│   └── utils/
│       ├── retry.py               # decorator: retry(max_retries, base_delay, jitter)
│       ├── io.py                  # 安全读写 / atomic write / jsonl append
│       └── seed.py
├── scripts/
│   ├── eval_one.sh                # 例：调起 CLI
│   └── leaderboard.sh
├── tests/
│   ├── test_parser.py
│   ├── test_adapters.py
│   ├── test_runner_resume.py
│   └── fixtures/
├── runs/                          # 运行产物（gitignore）
│   └── {run_name}_{timestamp}/
│       ├── run_config.yaml
│       ├── items.jsonl
│       ├── failed.jsonl
│       ├── summary.json
│       └── summary.md
├── requirements.txt
└── requirements-vllm.txt          # 可选附加依赖
```

---

## 8. 关键流程时序

### 8.1 API 模型 (并发) 主流程

```
CLI run --config foo.yaml
   │
   ▼
Config.load                 ──► merged dict（含 _base_）
   │
   ▼
DATASET_REGISTRY[name](cfg) ──► dataset 可迭代 Sample
MODEL_REGISTRY[name](cfg)   ──► model.load() (建立 OpenAI client / Gemini client)
   │
   ▼
Runner(dataset, model, cfg).run()
   │
   ├─ load items.jsonl → done set
   ├─ pending = [s for s in dataset if s.uid not in done]
   ├─ ThreadPoolExecutor(max_workers)
   │     for sample in pending:
   │         submit _infer_one(sample)
   │             ├─ model.build_messages(req)   # 抽帧 / 编码 / 拼 messages
   │             ├─ model.generate(req)         # 内部重试退避
   │             ├─ parse answer
   │             └─ return InferenceResult
   │     as_completed → append_jsonl + ProgressManager.update
   │     周期 _snapshot() 落盘
   │
   ▼
Reporter.aggregate(items.jsonl) → summary.json + summary.md
```

### 8.2 本地 vLLM (批处理) 主流程

```
... 同上 ...
Runner._run_batched:
   while pending:
       batch = pending[:B]
       reqs  = [InferenceRequest(s, ...) for s in batch]
       prompts = [model.build_messages(r) for r in reqs]    # 并行预处理（线程池）
       raws    = model.generate_batch(reqs)                 # vLLM 内部高吞吐
       for s, raw in zip(batch, raws):
           res = parse + judge
           append_jsonl(res)
       pending = pending[B:]
```

### 8.3 本地 Transformers (单进程) 主流程

```
Runner._run_sequential:
   for sample in tqdm(pending):
       res = _infer_one(sample)         # 串行；内部 try/except
       append_jsonl(res)
```

---

## 9. 重点设计决策与理由

1. **Sample 持有 MediaRef 而非加载好的字节**
   原因：本地模型直接接路径（HF processor 自带 IO）；API 模型才需要 base64 / 抽帧；按需加载避免大文件提前进内存。

2. **`is_thread_safe` 由模型自己声明**
   原因：本地 transformers/vllm 共享同一 GPU 状态，并发会出错；API 模型天然线程安全。Runner 据此自动选择 sequential / threaded / batched，用户无需关心。

3. **Resume 在 Runner 层做，不让 Model/Adapter 关心**
   原因：当前三个 Benchmark 各写了一套 resume 逻辑，重复且不一致。统一为 `items.jsonl` + `uid` 单一事实来源。

4. **抽帧、去音轨等媒体预处理统一到 `media/`**
   原因：Daily-Omni/OmniVideoBench 各自有 ffmpeg/cv2 代码；抽出来后被所有 API 模型复用。

5. **答案解析独立成 `eval/parser.py`**
   原因：三个 Benchmark 的字母抽取规则细节不一致（OmniBench 用 `index2ans` 全文匹配；Daily-Omni 用首字母 + `\b[ABCD]\b`；OmniVideoBench 用 JSON 解析 + `\boxed{}`）。统一成“多策略级联”，对所有模型与数据集都鲁棒。

6. **配置文件 + 注册表 ≫ Python 调用代码**
   原因：评测频繁、配置组合多（模型 × 数据集 × modality_mode × 并发），YAML 易复现易分享易做 leaderboard。

7. **OpenAI 兼容协议覆盖最广**
   原因：本地 vLLM 启动 OpenAI server (`vllm serve`) 也走同一个 `OpenAIChatModel`，无需新写适配；用户用 GPT-4o / DeepSeek / Qwen-Bailian / 自建 vLLM 都是同一份代码。

---

## 10. 与现有三个 Benchmark 的迁移路径

| 现有脚本 | 迁移后 |
|---|---|
| `Daily-Omni/test_model_api/main_tester.py` | `model=openai_chat` / `model=gemini` + `dataset=daily_omni` + `concurrency.mode=thread` |
| `Daily-Omni/test_model/Qwen2.5-Omni/testmodel.py` | `model=transformers_qwen25omni` 或 `model=vllm` + `dataset=daily_omni` |
| `OmniBench/inference/demo_api_call.py` | `model=openai_chat`（image_url 抽帧关闭）+ `dataset=omnibench` + `concurrency.mode=thread` |
| `OmniVideoBench/eval/gemini_eval.py` | `model=gemini` + `dataset=omnivideobench` + `concurrency.mode=thread` |
| `OmniVideoBench/eval/qwenomni_eval.py` | `model=transformers_qwen25omni` + `dataset=omnivideobench` |

迁移步骤：

1. 拷贝原 README 中 `--video_base_dir`/`--json_file_path`/`--input_mode` 这些参数 → 写成 YAML；
2. 第一阶段：保留原始三个仓库不动，Unify-OmniBench 通过 Adapter 直接读它们的原始数据文件；
3. 第二阶段：可选把原仓库中的本地推理脚本核心代码 import 进来（如 `process_mm_info`），避免重复实现。

---

## 11. 实施里程碑（建议）

| Phase | 内容 | 验收 |
|---|---|---|
| **M1 核心骨架（1-2 天）** | `core/types.py`、`registry`、`config`、`Runner._run_sequential`、`eval/parser.py`、`eval/metrics.py` | 跑通 1 个 dummy Adapter + 1 个 echo Model 的 e2e |
| **M2 数据适配（1 天）** | 三个 Adapter (`daily_omni` / `omnibench` / `omnivideobench`) | 三者 `len(dataset)` 与原仓库一致；随机抽 5 条字段对齐 |
| **M3 API 模型 + 并发（2 天）** | `openai_chat` + `gemini` + `Runner._run_threaded` + resume + 进度条 | 用 GPT-4o 跑 Daily-Omni 100 条，断点续跑可恢复 |
| **M4 本地模型（2-3 天）** | `transformers_qwen25omni` / `qwen25vl`，封装 `process_mm_info` | 复现 Daily-Omni README 中 Qwen2.5-Omni-3B 的精度 ±1% |
| **M5 vLLM + 批处理（1-2 天）** | `VLLMModel` + `_run_batched` | 与 M4 同模型精度对齐，吞吐 ≥ 3× |
| **M6 Reporter + Leaderboard（1 天）** | `summary.md` 表格、`unify-eval report` 多 run 对比 | 一键产出 README 中相同样式的表 |
| **M7 文档与示例（0.5 天）** | `README.md` quickstart、`ADAPTERS.md` 扩展指南、5 份典型 yaml | 新人按文档跑通 1 个模型 ≤15 分钟 |

---

## 12. 后续可扩展点

- **自由问答评估**：`eval/llm_judge.py`（GPT-4o-as-judge），适配 Daily-Omni 的开放生成模式。
- **Bootstrap CI / 子采样稳定性**：参考 Daily-Omni `run_subsampling_stability.sh`，做成 `unify-eval stability --run_dir ...`。
- **Agent 评测**：把 Daily-Omni 的 baseline (Qwen2-Audio 服务 + Qwen2.5-VL API + Qwen2.5 文本) 作为一个 `AgentModel` 接入。
- **多 GPU / 多机分片**：`Runner` 提供 `--shard i/N` 参数，配合外部调度（如 SLURM）切分 dataset。
- **HuggingFace datasets 一等公民**：`hf_loader` 直接 `load_dataset(repo_id, split)` 后用一个通用 `HFColumnMappingAdapter`（YAML 里声明列名映射）即可适配新数据集，无需写 Python。

# Unify-OmniBench

> 统一评测 **Daily-Omni / OmniBench / OmniVideoBench / WorldSense** 等多模态 Benchmark。
> 支持本地 `transformers` / `vLLM`，以及 OpenAI 兼容 / Gemini 外部 API。
> 自动并发/批处理、断点续跑、答案抽取、分桶报告。

## 支持的 Benchmark

| Benchmark | 题目数 | 模态 | 媒体 | 时长 | 题型数 | 音视频模式 (`use_audio_in_video`) | 来源 |
|---|---|---|---|---|---|---|---|
| **Daily-Omni** | 1197 | Video + Audio | .mp4 + .wav | 30s / 60s | 4 类 | 双路并列 (`false`) | — |
| **OmniBench** | 1142 | Image + Audio | .png/.jpg + .mp3 | — | 8 类 + 3 音频类型 | 双路并列 (`false`，无视频，不适用) | [🤗 m-a-p/OmniBench](https://huggingface.co/datasets/m-a-p/OmniBench) |
| **OmniVideoBench** | 1000 | Video + Audio | .mp4 (内嵌音轨) | 4s~32min | 13 类 + 3 音频类型 | 交织 (`true`) | [🤗 NJU-LINK/OmniVideoBench](https://huggingface.co/datasets/NJU-LINK/OmniVideoBench) |
| **WorldSense** | 3172 | Video + Audio | .mp4 (内嵌音轨) | <1min~>8min | domain/task_domain/audio_class 多维度 | 交织 (`true`) | [WorldSense](https://github.com/) (via VLMEvalKit) |

**音视频模式说明**（配置见 `unify_omnibench/config/dataset_config.yaml`，两种模式互不兼容）：
- **双路并列**：video/audio 作为两个独立的 `multi_modal_data` 条目发送，音频来自独立文件，无时间戳绑定。vLLM 长期支持，无版本限制。
- **交织**：音频从视频容器自身音轨按时间戳提取，与视频帧按 `seconds_per_chunk`（默认2秒）交替编码，不额外挂载独立audio文件。vLLM V1 曾一度不支持交织模式（[#25473](https://github.com/vllm-project/vllm/issues/25473)，[#33605](https://github.com/vllm-project/vllm/pull/33605) 于2026-02-04修复），当前 pin 的 `vllm==0.17.0`（见 `env_init.sh`）已验证生效（`tests/test_qwen_omni_vllm.py` 回归测试）；换更低版本 vllm 前建议先跑一下这个测试确认。Transformer 后端不受此限制。

---

## 目录

```
Unify-OmniBench/
├── run.py                       # CLI 入口
├── eval.sh                      # 一键脚本
├── docs/                        # 设计/架构/适配器文档
├── tests/
├── script/                      # 数据格式转换脚本
└── unify_omnibench/
    ├── runner.py                # 评测引擎 (sequential / thread / batch + resume)
    ├── core/                    # types / registry / config
    ├── prompt/                  # 统一媒体层 + prompt 模板
    ├── datasets/                # daily_omni / omnibench / omnivideobench / worldsense
    ├── models/                  # echo / openai_chat / openai_omni / gemini / qwen25omni / vllm
    ├── eval/                    # parser + report
    └── utils/
```

**核心设计**：Benchmark 与 Model 解耦，新增数据集只需适配器 + yaml。统一 `Sample` 数据模型，统一 `BaseModel.generate()` 接口，Runner 自动选择并发模式。

---

## 安装

```bash
pip install -e .              # 基础
pip install -e ".[api]"       # + openai / google-genai
pip install -e ".[local]"     # + torch / transformers
pip install -e ".[vllm]"      # + vllm
```

> Python ≥ 3.9

---

## 快速开始

```bash
# 编辑 eval.sh 顶部变量，然后：
bash eval.sh

# 或直接用 run.py：
python run.py --backend openai --dataset daily_omni \
    --model-name Qwen2.5-Omni-7B --api-url http://localhost:8001/v1

python run.py --backend vllm --dataset daily_omni \
    --model-path /path/to/Qwen2.5-Omni-7B --model-name Qwen2.5-Omni-7B
```

| 参数 | 说明 |
|---|---|
| `--backend` | `echo` / `openai` / `openai-omni` / `gemini` / `qwen_omni` / `vllm` |
| `--dataset` | `daily_omni` / `omnibench` / `omnivideobench` / `worldsense` |
| `--model-path` | 模型路径（本地 backend） |
| `--model-name` | 结果目录名 |
| `--api-url` | API 地址（API backend） |
| `--workers` | 并发数（默认 8） |
| `--resume` | 断点续跑 |
| `--rerun-failed` | 只重跑失败项 |

---

## Python API

```python
from unify_omnibench import build_dataset, build_model
from unify_omnibench.runner import Runner

cfg = {
    "run_dir": "runs/my_run", "modality_mode": "av",
    "dataset": {"name": "daily_omni", "qa_file": "...", "video_base_dir": "..."},
    "model":   {"name": "openai_chat", "model": "gpt-4o", "api_key": "sk-..."},
    "concurrency": {"mode": "thread", "max_workers": 8},
    "generation":  {"temperature": 0.0, "max_new_tokens": 16},
}
summary = Runner(build_dataset(cfg["dataset"]), build_model(cfg["model"]), cfg).run()
print(summary["accuracy"])
```

---

## 扩展新 Benchmark

1. 创建 `config/datasets/my_bench.yaml` 配置数据路径
2. 编写适配器继承 `BaseDatasetAdapter`，逐条产出 `Sample`
3. 运行 `BACKEND=vllm DATASET=my_bench bash eval.sh`

详见 `docs/ADAPTERS.md`。

---

## 测试

```bash
pytest tests/ -v
```

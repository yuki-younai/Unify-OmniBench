> ⚠️ **"1. 新增一个 Benchmark"一节描述的是早期设计（每个数据集单独一个
> Adapter 类），当前实际实现已统一成单一的 `unified.py::UnifiedAdapter` +
> `script/convert_*.py` 转换脚本**，新增数据集不需要再写 Python 适配器代码，
> 见 `README.md`「扩展新 Benchmark」一节。"2. 新增一个 Model"一节仍然适用。

# 扩展指南：新增 Benchmark / 新增 Model

本文档面向**贡献者**，描述如何把新数据集或新模型接入 Unify-OmniBench。

---

## 1. 新增一个 Benchmark (Dataset Adapter)

### 步骤

1. 在 `unify_omnibench/datasets/` 下新建 `my_bench.py`；
2. 继承 `BaseDatasetAdapter`，用 `@register_dataset("my_bench")` 注册；
3. 实现：
   - `__init__(self, cfg)`：从 `cfg` 拿数据路径、根目录、过滤条件等；
   - `__iter__(self)`：逐条 `yield Sample(...)`；
   - `__len__(self)`：用于进度条；
4. **必须保证 `Sample.uid` 全局唯一**（推荐 `self.make_uid(index, video_id, q_index)`）；
5. 在 `tests/test_adapters.py` 增加 5 条 fixture 用例；
6. 在 `configs/_base/` 增加一份 base yaml，列出该 dataset 必要字段；
7. 在 `unify_omnibench/datasets/__init__.py` `from . import my_bench`，触发注册。

### 字段映射小抄

| Sample 字段 | 必填？ | 来源建议 |
|---|---|---|
| `uid` | 必填 | `{dataset}:{idx}:{video_id_or_hash}` |
| `dataset` | 必填 | `self.name`（注册名） |
| `question` | 必填 | 原文 question |
| `choices` | 必填 | list[str]，可带前缀 "A." 也可不带 |
| `answer` | 评测必填 | `"A"/"B"/"C"/"D"`，统一成大写字母 |
| `media` | 视模型 | `[MediaRef("video"/"audio"/"image", path)]` |
| `meta` | 推荐 | 任何用于分桶的字段：`task_type / video_category / duration_s / audio_type / ...` |

### 答案归一化（重要！）

有些数据集 `answer` 是 `"correct_option"="B"`，有的是 `"answer"="B.xxx"`，有的是完整文本。Adapter **务必输出单字母**，否则 `eval/parser.py` 在精确匹配时会失败。如果原数据是文本，Adapter 内部要把它 map 回 choices 的字母。

---

## 2. 新增一个 Model

### 2.1 外部 API 模型（OpenAI 兼容协议）

**最佳路径：直接复用 `openai_chat`**，新增一份 YAML 即可，无需写代码：

```yaml
model:
  name: openai_chat
  api_key_env: MY_API_KEY
  base_url_env: MY_BASE_URL
  model: my-org/my-omni-model
  video: { seconds_per_frame: 1, max_frames: 16 }
```

只有当该 API **协议或多模态字段非 OpenAI 标准**时才需要写新 Model：

1. `unify_omnibench/models/api/my_api.py`；
2. 继承 `BaseModel`，`is_thread_safe = True`；
3. `generate(self, req)` 同步返回字符串；
4. 内部使用 `@retry(max_retries, base_delay)` 装饰底层 HTTP 调用。

### 2.2 本地 Transformers 模型

1. `unify_omnibench/models/local/my_model.py`；
2. 继承 `BaseModel`，`is_thread_safe = False`；
3. 在 `load()` 里建模型和 processor；
4. 在 `generate()` 中：
   - 用 `req.sample.media` + `req.modality_mode` 构造该模型族的 conversation；
   - 调用 processor / model.generate；
   - 返回解码后的字符串；
5. **优先复用** Daily-Omni 已有的 `test_model/<Model>/testmodel.py`：把它的核心函数（chat_template、process_mm_info、采样帧、generate）抽到 `unify_omnibench/media/<model>_utils.py`。

### 2.3 本地 vLLM 模型

1. 优先用通用 `vllm` 注册名 + 不同 YAML 覆盖；
2. 若 prompt 构造需特殊处理（多模态 placeholders），在 `models/vllm/` 下加子类 `VLLMQwenOmniModel(VLLMModel)` 覆写 `_build_prompt`；
3. `supports_batch = True`，Runner 会自动用 `_run_batched`。

---

## 3. 配置覆盖与命令行 override

约定：CLI 提供 `--set key.subkey=value` 多次（建议用 `OmegaConf` 实现，简单情形可自己解析）：

```bash
unify-eval run --config configs/qwen25omni_dailyomni_av.yaml \
    --set modality_mode=visual \
    --set generation.max_new_tokens=20 \
    --set concurrency.max_workers=8
```

---

## 4. 调试技巧

| 问题 | 排查方式 |
|---|---|
| 启动报 `Unknown dataset/model name` | 检查 `unify_omnibench/datasets/__init__.py` 或 `models/__init__.py` 是否 import 该子模块（注册必须在 import 时生效） |
| 跑 1000 条只评出 0% | 90% 是 `parse` 阶段失败 → 查看 `items.jsonl` 中 `raw_output` 与 `parsed_answer`；调 `extract_choice_letter` |
| 多线程跑 OpenAI 报 rate limit | 调小 `concurrency.max_workers`；模型层的 `retry` 已退避，但极端情形需限速 |
| 本地模型 OOM | 减小 `max_frames`；或换 `vllm` + `tensor_parallel_size>1` |
| Gemini files.upload 大文件失败 (500) | 模型层已捕获 500 跳过；该样本会写入 `failed.jsonl`，用 `unify-eval rerun-failed` 重投 |
| Resume 后没续上 | 确认 `items.jsonl` 中失败行的 `error` 字段非空（设计上失败不计入 done） |

---

## 5. 完整配置示例索引

| 文件 | 描述 |
|---|---|
| `configs/_base/dailyomni.yaml`        | Daily-Omni 通用 dataset + 默认 modality_mode=av |
| `configs/_base/omnibench.yaml`        | OmniBench 通用 dataset |
| `configs/_base/omnivideobench.yaml`   | OmniVideoBench 通用 dataset |
| `configs/qwen25omni_dailyomni_av.yaml`| 本地 Transformers Qwen2.5-Omni-3B，跑 Daily-Omni AV |
| `configs/qwen25omni_dailyomni_av_vllm.yaml` | 同上但走 vLLM 批处理 |
| `configs/gpt4o_omnibench_imgaudio.yaml` | GPT-4o（OpenAI 兼容）跑 OmniBench |
| `configs/gemini_omnivideobench.yaml`  | Gemini-2.5-Flash 跑 OmniVideoBench，多线程 |
| `configs/deepseek_text_dailyomni.yaml`| DeepSeek 纯文本（visual/audio 退化）跑 Daily-Omni |

每份 yaml 都遵循 `DESIGN.md §6.2` 的字段集，并通过 `_base_` 继承公共 dataset 块。

---

## 6. 完整 quickstart（实现完成后）

```bash
# 1) 安装
cd Unify-OmniBench
pip install -e .

# 2) 准备环境变量（API 模型）
export GPT4O_API_KEY=...
export GPT4O_BASE_URL=...
export GEMINI_API_KEY=...

# 3) 跑评测
unify-eval run --config configs/gpt4o_omnibench_imgaudio.yaml

# 4) 看结果
cat runs/gpt4o_omnibench_imgaudio_20260627-162300/summary.md

# 5) 失败项重投
unify-eval rerun-failed --run_dir runs/gpt4o_omnibench_imgaudio_20260627-162300/

# 6) 多模型横向对比
unify-eval report --runs runs/* --out leaderboard.md
```

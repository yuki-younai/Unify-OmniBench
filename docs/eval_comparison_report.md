# Unify-OmniBench vs 原始 Benchmark 评测流程对比报告

> 生成时间: 2026-07-06  
> 对比范围: Daily-Omni / OmniBench / OmniVideoBench × (Transformer / vLLM / OpenAI) 共 9 组合

---

## 总体结论

| Benchmark | Transformer | vLLM 离线 | OpenAI (vLLM serve) | 综合结论 |
|---|---|---|---|---|
| **Daily-Omni** | ✅ 高度一致（仅 processor 加载方式有微小差异） | ⚠️ 缺少长视频采样覆盖逻辑 | ⚠️ 原始实现面向外部 API，不可比对；prompt 不同 + 抽帧方式完全不同 | **Transformer 可作为对照基准; vLLM 缺长视频逻辑; OpenAI 全新实现** |
| **OmniBench** | ⚠️ prompt 文案不同 + max_tokens 不同(32→10) + processor 差异 | ⚠️ limit_mm_per_prompt 差异 + max_tokens | ❌ 无官方实现可比 → 只能交叉验证 | **prompt 差异是关键，需统一模板或保留原始模板** |
| **OmniVideoBench** | 🔴 **use_audio_in_video: True→False** + prompt 完全不同 + 答案解析方式不同 | ❌ 无官方 vLLM 实现 | ❌ 无官方实现可比 | **多项关键差异，需重点修复** |

---

## 一、Daily-Omni

### 1.1 Transformer 模式对比

| 维度 | 原始实现 (`testmodel.py`) | Unify-OmniBench (`qwen25omni.py`) | 是否一致 |
|---|---|---|---|
| **模型加载** | `from_pretrained(device_map="auto", dtype=bf16, attn_implementation="flash_attention_2", enable_audio_output=可配置)` | `from_pretrained(device_map="auto", torch_dtype=bf16, attn_implementation="flash_attention_2", enable_audio_output=False)` | ⚠️ `enable_audio_output` 原始默认为 True(加载TTS头), Unify 为 False。不影响文本输出结果 |
| **Processor 加载** | `Qwen2_5OmniProcessor.from_pretrained()` (无显式 use_fast) | `Qwen2_5OmniProcessor.from_pretrained(use_fast=True)` | ⚠️ Unify 显式 use_fast=True。官方文档说 fast/slow 输出有细微差异 |
| **System Prompt** | `"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group..."` | 完全一致 (`QWEN_OMNI_DEFAULT.system`) | ✅ |
| **User Prompt** | `"Your task is to accurately answer multiple-choice questions based on the {media_desc}. Select the single most accurate answer... Question: {q} Choices: {c} Your answer should be a capital letter... A, B, C, or D. Don't generate any other text."` | 完全一致 (`_USER_PROMPT_NORM`) | ✅ |
| **media_desc** | `"given video and audio together"` / `"given video"` / `"given audio"` | 完全一致（通过 `media_description()` 生成） | ✅ |
| **Content 顺序** | `video → audio → text`（all 模式） | `filter_media()` 返回介质顺序与 Sample.media 一致 → `video → audio → text` | ✅ |
| **use_audio_in_video** | 硬编码 `False`（`get_effective_use_audio_in_video`） | 硬编码 `False`（`build_messages` 返回 `False`） | ✅ |
| **max_new_tokens 默认** | 10 | 10 | ✅ |
| **do_sample** | `False` | `False` | ✅ |
| **num_beams** | `1` | `1` | ✅ |
| **eos_token_id** | `processor.tokenizer.eos_token_id` | `self.processor.tokenizer.eos_token_id` | ✅ |
| **return_audio** | `False` | `False` | ✅ |
| **process_mm_info 来源** | `qwen_omni_utils.process_mm_info` | `qwen_omni_utils.process_mm_info` | ✅ |
| **张量搬运** | `is_floating_point() → device+dtype`, 整形 → device | 完全一致 | ✅ |
| **答案解析** | `extract_choice_letter()`: 首字符+独立字母正则 | `eval/parser.py::extract_choice_letter()`: 6级级联（首字符优先级保留） | ✅（Unify 是超集） |

**结论**: 几乎完全一致。仅 `enable_audio_output` 和 `use_fast` 有形式上的差异，不影响精度。

---

### 1.2 vLLM 离线模式对比

| 维度 | 原始实现 (`testmodel.py` `--use_vllm`) | Unify-OmniBench (`vllm_runner.py`) | 是否一致 |
|---|---|---|---|
| **Prompt/消息构造** | 同 Transformer 的 `build_conversation()` | 复用 `qwen25omni.py::build_messages()` | ✅ |
| **use_audio_in_video** | `False` | `False` | ✅ |
| **LLM 参数** | | | |
| - `trust_remote_code` | `True` | `True` | ✅ |
| - `tensor_parallel_size` | 从 `CUDA_VISIBLE_DEVICES` 自动检测 | 同逻辑 (`_detect_gpu_count()`) | ✅ |
| - `gpu_memory_utilization` | `0.95` (默认) | `0.95` (默认) | ✅ |
| - `max_num_seqs` | `1` (默认) | `1` (默认) | ✅ |
| - `max_model_len` | `32768` (默认) | `32768` (默认) | ✅ |
| - `limit_mm_per_prompt` | `{"image":1,"video":1,"audio":1}` | `{"image":1,"video":1,"audio":1}` | ✅ |
| - `seed` | ❌ 未设置 | `1234` | ⚠️ 细微差异（原始无 seed） |
| - `enforce_eager` | ❌ 未设置 | `True` | ⚠️ 细微差异 |
| - `mm_processor_cache_gb` | ❌ 未设置（默认 4GB） | `0` (显式关闭) | ⚠️ Unify 修复了 vllm 0.11.0 bug |
| **SamplingParams** | | | |
| - `temperature` | `0.0` | `0.0` | ✅ |
| - `top_p` | `1.0` | `1.0` | ✅ |
| - `top_k` | `-1` | `-1` | ✅ |
| - `max_tokens` | `args.max_new_tokens` (默认 10) | `max_new_tokens` (默认 10) | ✅ |
| **vLLM 输入构造** | | | |
| - `prompt` | `processor.apply_chat_template(...)` | 同 | ✅ |
| - `multi_modal_data` | `{"image": images, "video": videos, "audio": audios}` (保持 list) | 单元素 `images[0]` 等 unwrap（对齐官方示例） | ⚠️ 形式差异 |
| - `mm_processor_kwargs` | `{"use_audio_in_video": False}` | `{"use_audio_in_video": False}` | ✅ |
| **环境变量** | `VLLM_USE_V1=0` (vLLM 旧版本兼容) | 不设 `VLLM_USE_V1` (vllm 0.11.0 V0 已移除) | ⚠️ 版本适配差异 |
| **🆕 长视频采样覆盖(60s)** | ✅ `vllm_long_video_fps=1.0, min_frames=4, max_frames=192` | ❌ **未实现** | 🔴 **关键差异** |

**结论**: 

- 大部分参数一致，但 **长视频(60s)采样覆盖逻辑未移植**，这是影响精度的关键差异。原始实现在 60s 视频上使用 1fps 而非默认 2fps，抽取帧数上限从 768 降至 192。
- `multi_modal_data` 的 list→bare 解包是形式差异（Unify 对齐 vLLM 官方示例），实际效果应等价。
- `mm_processor_cache_gb=0` 和 `enforce_eager=True` 是 Unify 为解决 vllm 0.11.0 稳定性问题而加的安全措施，不影响正常情况下的结果。

---

### 1.3 OpenAI 兼容 API 模式对比

| 维度 | 原始实现 (`test_model_api/`) | Unify-OmniBench (`openai_chat.py`) | 是否一致 |
|---|---|---|---|
| **目标模型** | Gemini/GPT-4o/DeepSeek **外部 API** | **本地 vLLM serve 部署的 Qwen2.5-Omni** | 🔴 不可比对 |
| **System Prompt** | `"Your task is to accurately answer multiple-choice questions based on the given video..."` | `"You are a multimodal evaluator. Answer with one letter only (A/B/C/D)."` | 🔴 不同 |
| **User Prompt** | `"Given the video, answer the question below.\nQuestion: {q}\nChoices: {c}"` | `_USER_PROMPT_NORM` (同 transformer 路径) | 🔴 不同 |
| **视频处理** | `cv2` 逐帧读取, fps≈0.5 (seconds_per_frame=2), 无 max_frames 上限 | 多种模式，默认 `qwen_native`: 用 `qwen_omni_utils.fetch_video` 对齐 transformer 路径 | 🔴 完全不同 |
| **音频处理** | 原始 base64 编码 (无重采样) | 16kHz librosa 重采样后 base64 | 🔴 完全不同 |
| **重试策略** | 固定 4 次, 指数退避 base_delay=4 | 同逻辑（通过 `utils/retry.py`） | ✅ |
| **max_tokens** | 10 (OpenAI 兼容路径) | 10 (默认) | ✅ |

**结论**: 原始 Daily-Omni 的 OpenAI 路径面向的是闭源外部 API，而非本地部署的 Qwen2.5-Omni 走 vLLM serve 协议。Unify 的 `openai_chat` 后端是全新实现，专门针对本地 vLLM serve + Qwen2.5-Omni 场景。两者目标不同，无法直接比对。**应以 Unify 内部的 transformer 后端结果作为 openai 后端的对照基准**（通过 prompt/抽帧一致性保证等价性）。

---

## 二、OmniBench

### 2.1 Transformer 模式对比

| 维度 | 原始实现 (`local_inference.py`) | Unify-OmniBench (`qwen25omni.py`) | 是否一致 |
|---|---|---|---|
| **模型加载** | `from_pretrained(device_map="auto", torch_dtype=bf16, attn_implementation="flash_attention_2")` | 同 | ✅ |
| **model.eval()** | ✅ 显式调用 | ❌ 未调用 | ⚠️ 细微差异 |
| **Processor** | `Qwen2_5OmniProcessor.from_pretrained()` | `Qwen2_5OmniProcessor.from_pretrained(use_fast=True)` | ⚠️ use_fast 差异 |
| **System Prompt** | `"You are Qwen, a virtual human..."` | 同 | ✅ |
| **User Prompt** | `"Please answer the following question based on the given image and audio:\nQuestion: {q}\nOptions:\n{options_text}\nAnswer with a single capital letter: A, B, C, or D."` | `"Your task is to accurately answer multiple-choice questions based on the given image and audio together.\nSelect the single most accurate answer...\nQuestion: {q}\nChoices: {c}\nYour answer should be a capital letter... A, B, C, or D. Don't generate any other text."` | 🔴 **关键差异** |
| **选项格式** | `"A. xxx\nB. xxx"` (原始`options`为list，每个已带字母前缀) | `"\n".join(choices)` (可能不带前缀) | ⚠️ 取决于数据格式 |
| **Content 顺序** | `image → audio → text`（条件判断存在性） | `filter_media()`: image → audio → text | ✅ |
| **use_audio_in_video** | `False` | `False` | ✅ |
| **max_new_tokens 默认** | **32** | **10** | 🔴 **关键差异** |
| **do_sample** | `False` | `False` | ✅ |
| **num_beams** | `1` | `1` | ✅ |
| **答案解析** | `extract_choice_letter()` (首字符+独立字母) + `map_answer_to_letter()` 完整文本→字母映射 | `extract_choice_letter()` 6级级联 + `choices_to_index2ans()` | ⚠️ 形式差异；`map_answer_to_letter` 可能处理原始未妥善 |
| **断点续跑** | 读 JSONL 按 index 去重 | 读 items.jsonl 按 uid 去重 | ✅ 功能等价 |
| **批处理** | 逐条处理（单线程循环） | `generate_batch()` 批量推理 | ⚠️ Unify 支持 batch |

**结论**:

| 关键差异 | 影响评估 |
|---|---|
| **User Prompt 措辞不同** | 🔴 高影响。原始 prompt 更简短、指定了 "based on the given image and audio"（而非通用的 `media_desc` 占位），对只含单个媒体的情况措辞不同 |
| **max_new_tokens: 32→10** | 🔴 高影响。32 可以容纳选项内容全文输出（需要反向查找解析），10 仅够输出单字母。这意味着 Unify 默认配置下模型输出的长文本可能被截断 |
| **选项格式 `A. xxx` vs 纯文本** | ⚠️ 取决于数据源，可能无差异 |
| **解析策略差异（map_answer_to_letter）** | ⚠️ 中影响。原始用 `map_answer_to_letter(answer_text, options)` 直接匹配原始答案文本到字母；Unify 的 `choices_to_index2ans` 可能对 `"A. xxx"` 格式的 choices 切割不同 |

---

### 2.2 vLLM 离线模式对比

| 维度 | 原始实现 (`local_inference.py --use_vllm`) | Unify-OmniBench (`vllm_runner.py`) | 是否一致 |
|---|---|---|---|
| **以上所有 Prompt 差异** | 同上 | 同上 | 🔴 同上 |
| **max_new_tokens** | 32 (从 args) | 10 (默认) | 🔴 同上 |
| **LLM 参数** | | | |
| - `limit_mm_per_prompt` | `{"image": 1, "video": **0**, "audio": 1}` | `{"image": 1, "video": 1, "audio": 1}` | ⚠️ 细微差异（OmniBench 无视频，设置 0 不影响功能） |
| - `seed` | ❌ 未设置 | `1234` | ⚠️ |
| - `enforce_eager` | ❌ 未设置 | `True` | ⚠️ |
| - `mm_processor_cache_gb` | ❌ (默认 4GB) | `0` | ⚠️ |
| **SamplingParams** | `temperature=0.0, max_tokens=32` | `temperature=0.0, top_p=1.0, top_k=-1, max_tokens=10` | ⚠️ top_p/top_k 显式设置（原始用默认值） |
| **vLLM 输入** | `{"prompt":..., "multi_modal_data":{...}, "mm_processor_kwargs":{"use_audio_in_video":False}}` | 同，但 multi_modal_data 单元素 unwrap | ⚠️ 形式差异 |

**结论**: 除与 Transformer 共通的 Prompt + max_tokens 差异外，vLLM 特有差异均为安全参数和形式差异，不影响核心行为。

---

### 2.3 OpenAI 兼容 API 模式对比

**结论**: 🔴 **无原始官方实现可比**。OmniBench 原始的 `demo_api_call.py` + `closed_source_model.py` 面向 GPT-4o/Claude/Gemini 等闭源 API，而非本地 vLLM serve 的 Qwen。Unify 的 `openai_chat` 后端是全新实现的用法。**应以 Unify 内部 transformer 结果作为 openai 后端的对照基准**。

额外注意事项：
- 原始 OmniBench 使用 `parse_multi_choice_response()` 解析答案（更复杂的级联策略），而 Unify 使用统一的 `extract_choice_letter()`。两者可能对相同模型输出产生不同的解析结果，影响精度对比。

---

## 三、OmniVideoBench

### 3.1 Transformer 模式对比

这是**差异最大的一个组合**。

| 维度 | 原始实现 (`qwenomni_eval.py`) | Unify-OmniBench (`qwen25omni.py`) | 是否一致 |
|---|---|---|---|
| **模型加载** | `from_pretrained(torch_dtype=bf16, device_map="auto", attn_impl="flash_attention_2")` | 同 + `enable_audio_output=False` | ⚠️ 形式差异 |
| **disable_talker()** | ✅ `model.disable_talker()` | ❌ 使用 `enable_audio_output=False` (等价) | ✅ 功能等价 |
| **Processor** | `Qwen2_5OmniProcessor.from_pretrained()` | `use_fast=True` | ⚠️ 细微差异 |
| **System Prompt** | `"You are Qwen, a virtual human..."` | 同 | ✅ |
| **User Prompt** | `"You are given a video. Based on the content of the video, answer the following question:\n\nQuestion:\n{q}\n\nOptions:\n{options_text}\n\nAnswer with the option's letter directly(e.g., A, B, C, or D).If your access to the video content is limited, at least one option that is more likely than the others must be chosen.Mustn't give any other reason for can not choose!"` | `_USER_PROMPT_NORM`: `"Your task is to accurately answer multiple-choice questions based on the given video.\nSelect the single most accurate answer...\nQuestion: {q}\nChoices: {c}\nYour answer should be a capital letter... Don't generate any other text."` | 🔴 **关键差异** |
| **use_audio_in_video** | 🔴 **`True`** (硬编码) | 🔴 **`False`** (硬编码 + dataset_config 覆盖为 false) | 🔴🔴 **最关键差异** |
| **音频处理** | 自定义 `utils/audio_process.py`: `librosa.load(sr=16000)` + `assert` 视频有音轨 | 使用 `qwen_omni_utils.process_mm_info` | 🔴 **关键差异** |
| **视频处理** | 自定义 `utils/vision_process.py` | 使用 `qwen_omni_utils.vision_process` | ⚠️ 可能不同 |
| **帧数上限** | `max_frames=256` 均匀降采样 | 委托给 `qwen_omni_utils` 内部处理 (默认 768) | ⚠️ **关键差异** |
| **max_new_tokens 默认** | **64** | **10** | 🔴 **关键差异** |
| **do_sample** | `False` (默认，仅 `--do_sample` 时才开启) | `False` | ✅ |
| **temperature** | `0.7` (默认，仅 do_sample=True 时使用) | 不传（do_sample=False 时忽略） | ✅ |
| **batch_size** | 默认 `4` 条/批 | 通过 `workers` 参数控制（默认 8） | ⚠️ 可能不同 |
| **答案解析** | `extract_model_answer()`: 去除 `assistant` 标记 → 解析 `/box{X}` / `\boxed{X}` → `clean_text()` 小写比较 | `extract_choice_letter()`: 6级级联 → 字母比较 | 🔴 **关键差异** |
| **答案判对** | `clean_text(model_answer) == clean_text(correct_answer)` (文本内容比对) | `parsed_letter == sample.answer` (字母比对) | 🔴 **关键差异** |

**结论 - 高影响差异汇总**:

| 差异点 | 影响评估 |
|---|---|
| **`use_audio_in_video: True→False`** | 🔴🔴 最高影响。原始实现将音频按时间戳与视频帧交织编码（官方论文的设计），Unify 将音频/视频分开编码。这直接改变模型感知到的时间-音频对应关系 |
| **Prompt 完全不同** | 🔴 高影响。原始的 "You are given a video" 措辞与 "Your task is to accurately answer" 可能引导模型不同的输出格式，尤其原始额外要求了 "at least one option..." 等兜底逻辑 |
| **max_new_tokens: 64→10** | 🔴 高影响。64 允许模型输出较长的思考文本再给答案；10 只够输出字母 |
| **帧数上限 256→768** | ⚠️ 中影响。更多帧数增加编码开销和内存，但提供更多视觉信息 |
| **答案解析方式不同** | 🔴 高影响。原始用文本内容比对（`clean_text()`），Unify 用字母比对（`parsed_letter == sample.answer`）。如果模型输出的是选项文本而非字母，原始可以正确评判，Unify 可能解析失败 |
| **音频处理路径不同** | ⚠️ 中影响。自定义 `audio_process.py` vs `qwen_omni_utils` 可能有实现差异 |

**特别说明**: `dataset_config.yaml` 中 `omnivideobench.use_audio_in_video` 被注释解释为 **当前临时设为 False**（因为 vLLM serve 不稳定），但原始官方实现是 `True`，这明确标记了一个已知但尚未解决的不一致。

---

### 3.2 vLLM 离线模式对比

**结论**: 🔴 **原始官方没有 vLLM 离线推理脚本**。OmniVideoBench 原始仓库中没有任何 `from vllm import LLM` 的代码（仅在 `environment_ming.yml` 里列为依赖，但无调用）。因此 Unify 的 vLLM 后端对 OmniVideoBench 是全新实现，无原始对照。**应以 Unify 内部的 transformer 结果作为 vLLM 后端的交叉验证基准**。

额外注意：OmniVideoBench 的 transformer 路径已与原始有 `use_audio_in_video` 差异，所以 transformer 结果本身未必等于"官方精度"。

---

### 3.3 OpenAI 兼容 API 模式对比

**结论**: 🔴 **无官方实现可比**。原始 `deepseek_eval.py` 面向 DeepSeek 纯文本 API，`gemini_eval.py` 面向 Google Gemini。两者都不直接对应本地 Qwen2.5-Omni 走 vLLM serve 的场景。Unify 的 `openai_chat` 后端是全新实现。**应以 Unify 内部 transformer 结果作为对照基准**。

---

## 四、通用对比：跨 Benchmark 共用组件

### 4.1 答案解析 (`parser.py`)

| 维度 | 原始实现（各 benchmark 不同） | Unify-OmniBench | 差异 |
|---|---|---|---|
| **Daily-Omni** | `extract_choice_letter()`: 首字符 → 独立字母正则 | 6级级联: JSON→boxed→括号→首字符→反查→独立字母 | ✅ Unify 是超集，兼容性更好 |
| **OmniBench** | `extract_choice_letter()` + `map_answer_to_letter()` + `parse_multi_choice_response()` | 同上 6级级联 + `choices_to_index2ans()` | ⚠️ `parse_multi_choice_response` 对多候选、括号、格式变化处理更复杂 |
| **OmniVideoBench** | `extract_model_answer()`: 去 assistant 标记→解析 boxed→`clean_text()` 小写比较 | 6级级联字母比对 | 🔴 原始用文本内容比对，Unify 用字母比对，本质不同 |

### 4.2 视频抽帧参数

| 参数 | Daily-Omni 原始 Transformer | Unify Transformer | Daily-Omni 原始 vLLM(长视频) | Unify vLLM | Unify OpenAI(qwen_native) |
|---|---|---|---|---|---|
| **默认 fps** | 委托 `qwen_omni_utils` (2fps) | 同 (2fps) | 2fps / **1fps (60s视频)** | 2fps | 2fps (通过 `fetch_video`) |
| **max_frames** | 委托内部 (768) | 同 (768) | 768 / **192 (60s)** | 768 | 768 |
| **min_frames** | 委托内部 (4) | 同 (4) | 4 | 4 | 4 |
| **resize/pixel** | `qwen_omni_utils` 内部 | 同 | 同 | 同 | 同 (通过 `fetch_video`) |

### 4.3 use_audio_in_video 汇总

| Benchmark | 原始 Transformer | 原始 vLLM | Unify Transformer | Unify vLLM | Unify OpenAI | 备注 |
|---|---|---|---|---|---|---|
| **Daily-Omni** | `False` | `False` | `False` | `False` | `False` (默认) | ✅ 一致 |
| **OmniBench** | `False` | `False` | `False` | `False` | `False` (默认) | ✅ 一致（OmniBench无视频） |
| **OmniVideoBench** | 🔴 `True` | ❌ 无实现 | 🔴 `False` | 🔴 `False` | 🔴 `False` (默认) | 🔴 所有 Unify 路径与原始不一致 |

---

## 五、按维度汇总差异清单

### 🔴 高影响差异（可能直接影响评测精度）

| # | 差异项 | 影响范围 | 详情 |
|---|---|---|---|
| 1 | **OmniVideoBench `use_audio_in_video`** | OmniVideoBench 全部 Unify 后端 | `True→False`，改变音频-视频时间对齐方式 |
| 2 | **OmniVideoBench User Prompt** | OmniVideoBench 全部 Unify 后端 | 原始 prompt 完全被替换为通用模板 |
| 3 | **OmniVideoBench 答案解析** | OmniVideoBench 全部 Unify 后端 | 文本内容比对→字母比对 |
| 4 | **OmniVideoBench max_new_tokens** | OmniVideoBench 全部 Unify 后端 | 64→10 |
| 5 | **OmniBench max_new_tokens** | OmniBench 全部 Unify 后端 | 32→10 |
| 6 | **OmniBench User Prompt 差异** | OmniBench 全部 Unify 后端 | "Please answer based on the given image and audio" vs 通用模板 |
| 7 | **Daily-Omni vLLM 长视频采样覆盖** | Daily-Omni Unify vLLM | 60s 视频 fps 1.0→2.0, max_frames 192→768 |

### ⚠️ 中影响差异（可能影响部分样本或特定场景）

| # | 差异项 | 影响范围 | 详情 |
|---|---|---|---|
| 8 | **OmniVideoBench 帧数上限** | OmniVideoBench Unify | 256→768 (qwen_omni_utils 默认) |
| 9 | **Processor `use_fast=True`** | 所有 Unify Transformer/vLLM | 官方说 fast/slow 图像预处理有细微差异 |
| 10 | **OmniBench `map_answer_to_letter` vs `choices_to_index2ans`** | OmniBench 数据加载 | 原始答案为文本时的字母映射可能不同 |

### ✅ 低影响差异（安全参数/形式差异，基本不影响精度）

| # | 差异项 | 详情 |
|---|---|---|
| 11 | `enable_audio_output=False` vs `disable_talker()` | 功能等价（仅评测模式不加载 TTS 头） |
| 12 | vLLM `seed=1234` | 原始未设置，但 temperature=0 时无影响 |
| 13 | vLLM `enforce_eager=True` | 安全措施，不影响结果 |
| 14 | vLLM `mm_processor_cache_gb=0` | 修复 vllm 0.11.0 bug，不影响正常结果 |
| 15 | vLLM `multi_modal_data` list→bare unwrap | 形式差异，对齐官方示例 |

---

## 六、建议修复优先级

### P0 — 必须修复（直接影响评测可比性）

1. **OmniVideoBench `use_audio_in_video`** → 改为 `True`（对齐官方实现）
   - 需要确认 vLLM serve / openai 后端下 `True` 是否稳定可用
   - 如果不可用，至少 transformer 后端应为 `True`

2. **OmniVideoBench 保留原始 Prompt** → 在 `dataset_config.yaml` 添加 `prompt_template` 覆盖
   ```yaml
   omnivideobench:
     prompt_template: |
       You are given a video. Based on the content of the video, answer the following question:
       
       Question:
       {question}
       
       Options:
       {choices}
       
       Answer with the option's letter directly(e.g., A, B, C, or D).If your access to the video content is limited, at least one option that is more likely than the others must be chosen.Mustn't give any other reason for can not choose!
   ```

3. **OmniVideoBench max_new_tokens** → 改为 64（或在 `eval.sh` 中通过 `MAX_NEW_TOKENS=64` 覆盖）

### P1 — 应该修复（影响评测排名的公平性）

4. **OmniBench 保留原始 Prompt** → 添加 `prompt_template` 覆盖
5. **OmniBench max_new_tokens** → 改为 32（或在运行时通过 `--max-new-tokens 32` 覆盖）
6. **Daily-Omni vLLM 长视频采样覆盖** → 移植到 `vllm_runner.py`

### P2 — 建议优化（提升准确性）

7. **OmniVideoBench 答案解析** → 增加文本内容比对路径（模拟 `clean_text()`）
8. **OmniVideoBench 帧数上限** → 统一为 256（与原始一致）
9. **OmniBench 答案映射** → 确认 `choices_to_index2ans` 与 `map_answer_to_letter` 等价或增加兼容

---

## 七、补充说明

### 7.1 关于 "无官方对照" 的组合

以下组合没有原始官方实现可比，只能通过 Unify 内部交叉验证：

- OmniBench × OpenAI (vLLM serve)
- OmniVideoBench × vLLM 离线
- OmniVideoBench × OpenAI (vLLM serve)

对于这些组合，建议：
1. 确保与同 benchmark 的 transformer 后端使用完全相同的 prompt 和预处理参数
2. 对比三种后端之间的精度差异（transformer vs vLLM vs openai），差异应 < 0.5%
3. 如果差异 > 1%，检查对应后端的特殊处理逻辑

### 7.2 关于 OpenAI 后端的 `qwen_native` 视频模式

Unify 的 `openai_chat.py` 提供了 `qwen_native` 视频模式，通过 `qwen_omni_utils.fetch_video` 精确复制 transformer 路径的抽帧逻辑（包括 pixel budget resize），这是实现 transformer ↔ openai 结果一致的**关键设计**。使用此模式时，应确保服务器端不覆盖 `mm_processor_kwargs`。

### 7.3 关于 eval.sh 中的 `MAX_NEW_TOKENS=512`

当前 `eval.sh` 第 40 行设置 `MAX_NEW_TOKENS=512`，这会覆盖所有 benchmark 的默认值。这意味着实际运行时 OmniBench 和 OmniVideoBench 的 max_tokens 差异（32 vs 64 vs 10）在这个设置下会被统一为 512，实际上隐藏了这些差异。建议：
- 对 OmniBench 使用 `MAX_NEW_TOKENS=32`
- 对 OmniVideoBench 使用 `MAX_NEW_TOKENS=64`
- 评估是否需要区分设置

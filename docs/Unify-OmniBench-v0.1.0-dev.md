# Unify-OmniBench v0.1.0 踩坑记录

> 记录 v0.1.0 开发/调试过程中遇到的、非显而易见的坑，方便日后升级依赖或排查
> 类似问题时快速定位，不用重新走一遍完整排查流程。按模块分组，每条给出：
> 现象 → 根因 → 修复/结论 → 涉及文件。

---

## 1. vLLM 多模态处理器缓存对重复视频返回 `None` 占位符

**现象**：同一段视频内容第 2 次及以后被不同题目请求时，偶发
`TypeError: 'NoneType' object is not subscriptable`
（出现在 `vllm/model_executor/models/qwen2_5_omni_thinker.py::_maybe_apply_prompt_updates`）。

**根因**：vLLM 的多模态**处理器**缓存 `mm_processor_cache_gb`（默认 4GB，**默认开启**！
跟 KV-cache 的 `enable_prefix_caching` 是完全独立的两套缓存）按内容哈希缓存已处理过的
多模态特征。同一视频哈希第二次命中时返回 `None` 占位表示"复用缓存"，而
pinned 版本的 `use_audio_in_video` 自动检测代码没有对这个 `None` 做判空保护。

**修复**：`config/models/vllm.yaml::mm_processor_cache_gb: 0` 直接关闭这套缓存
（这里每条样本单独调 `generate()`，同一视频本来就会被多个题目复用，关掉缓存
性能损失可忽略）。

**涉及文件**：`unify_omnibench/config/models/vllm.yaml`、
`unify_omnibench/models/vllm_backend/vllm_runner.py`

---

## 2. vLLM V1 引擎的交织音频（`use_audio_in_video=True`）曾经必现崩溃

**现象**：`use_audio_in_video=True` 时，任何长度超过一个 chunk 周期（默认2秒）的
视频，必现：
```
RuntimeError: Worker failed with error 'index 1 is out of bounds for dimension 0 with size 1'
```
根源是 `vllm/model_executor/layers/rotary_embedding/mrope.py` 里 mRoPE 位置计算
把"第几个音视频交织 chunk"错误地当成了"第几个视频"去索引 `video_grid_thw`。

**根因**：这是 vLLM 自己缺失的功能，不是配置问题——V1 引擎在 pinned 版本
（vllm==0.11.0，2025-10 发布）上压根没实现 Qwen2.5-Omni 的交织支持。上游
issue [#25473](https://github.com/vllm-project/vllm/issues/25473)、专项修复
[PR #33605](https://github.com/vllm-project/vllm/pull/33605)（2026-02-04 合入）。

**修复**：升级 vllm 到 `0.17.0`（见 `env_init.sh`，晚于上述修复合入时间足够久），
用 `tests/test_qwen_omni_vllm.py` 的交织回归测试验证（同一视频/shape，之前必现
崩溃，升级后 5/5 通过）。**每次升级 vllm 后建议重跑一次这个回归测试**，确认
交织模式没有回归。

**涉及文件**：`tests/test_qwen_omni_vllm.py`、`env_init.sh`

---

## 3. 长视频不限制帧数会打爆显存（`EngineDeadError`）

**现象**：`fps=2.0` 且没设 `max_frames` 上限时，长视频（WorldSense 最长档 >8min）
在 `use_audio_in_video=True` 下会解码出远超正常预算的帧数，把显存打爆，
vLLM 的 GPU worker 子进程直接崩掉（`EngineDeadError`），且**引擎崩溃后不会
自动恢复**——之后所有请求都会立刻报同一个错，看起来像是"越来越多样本失败"。

**修复**：给每个数据集的 `video:` 配置显式设置 `max_frames`（`dataset_config.yaml`），
在解码源头就限制帧数预算，而不是解码完默认上限（`qwen_omni_utils` 默认
`FPS_MAX_FRAMES=768`）再事后裁剪。

**涉及文件**：`unify_omnibench/config/dataset_config.yaml`

---

## 4. Transformer 后端意外加载了完整 TTS 流水线，看起来像卡死

**现象**：`model.generate()` 在生成完文本 token 之后似乎"卡住不动"，实际上是还在
跑 Talker 自回归生成语音 token + Token2Wav 的 DiT 声码器合成波形——因为
`from_pretrained()` 没传 `enable_audio_output=False`，完整加载了 Talker+Token2Wav；
`generate()` 也没传 `return_audio=False`。

**修复**：`enable_audio_output=False`（加载时跳过 TTS 模块）+ `return_audio=False`
（生成时只要文本），两处都要设置，缺一个仍然会触发 TTS 流程。

**涉及文件**：`unify_omnibench/models/local/qwen25omni.py`

---

## 5. `concurrency_mode`/`max_num_seqs` 配置死锁，`WORKERS` 参数完全被忽略

**现象**：`eval.sh` 里设了 `WORKERS=16`，但 vLLM 后端实际跑起来永远是逐条串行
（`Concurrency mode = sequential`），无论 `WORKERS` 设多大都没有任何效果。

**根因**：`Runner._resolve_mode()` 只要 `concurrency.mode != "auto"` 就直接用配置
值、完全不看 `model.supports_batch`；而 `vllm.yaml::concurrency_mode` 曾长期硬编码
为 `sequential`（因为当时 `generate_batch()` 的实现只是 Python 层 for 循环单条
调用，跟 sequential 完全等效，故意设成 sequential 更直观）。后来把
`generate_batch()` 改成真正的多 prompt 一次性提交后，忘了同步把
`concurrency_mode` 切到 `batch`、也忘了把 `max_num_seqs`（从 1）调大——
`max_num_seqs` 上限了 vLLM 引擎内部**真实并发调度数**，就算一次提交了 N 条
prompt，`max_num_seqs=1` 时引擎内部依然一条条串行处理。

**修复**：
- `vllm.yaml::concurrency_mode: batch`
- `run.py` 里把 `--workers`/`WORKERS` 直接覆盖进 `model_cfg["max_num_seqs"]`，
  跟 `concurrency.batch_size` 天然保持同一个值，不用手动同步两份配置

**涉及文件**：`unify_omnibench/config/models/vllm.yaml`、`run.py`、
`unify_omnibench/runner.py`

---

## 6. WorldSense 的 `use_audio_in_video` 该设 True 还是 False——别只看数据集"音频是否内嵌"

**现象**：WorldSense 的音频是内嵌在 `.mp4` 容器里的，直觉上"天然适合交织"，
于是想当然地把 `use_audio_in_video` 设成 `True`。

**教训**：数据的存储形式（内嵌 vs 独立文件）跟推理时**该不该用交织模式**是两件
独立的事——决定性因素是**参考实现（VLMEvalKit）实际怎么跑的**，不是数据长什么样。
去读 `VLMEvalKit/eval.sh` 生成的 config 文件，第一次囫囵扫过去漏看了关键两行
（`"use_audio_in_video": true`），得出"参考实现是双路并列"的错误结论；仔细重读
才发现参考实现其实是**交织 + 独立音频文件同时挂**（模型把同一段音频听两遍）。

**结论（v0.1.0 采用的简化版）**：为避免"同一段音频喂两遍"的冗余复杂度，
WorldSense 最终选择**只用交织**（`use_audio_in_video: true`，不挂独立 audio），
不追求跟参考实现逐字节对齐——因为这个"双路+交织同时挂"的组合本身也没有专门
测过在 vLLM 上是否稳定，简化实现的收益大于精确复现参考实现的收益。

**教训总结**：读第三方参考实现的配置生成代码时，**一定要读全**（哪怕看起来
像是无关的几行），一次遗漏足以得出完全相反的结论。

**涉及文件**：`unify_omnibench/config/dataset_config.yaml`、
`script/convert_worldsense.py`

---

## 7. transformers/vllm 版本不兼容的兼容性 shim

**现象**：`AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`，
出现在主进程（`LLMEngine.__init__`）和 vLLM spawn 出的 worker 子进程两处。

**根因**：为支持 Qwen2.5-Omni 而装的较新 transformers 版本移除了这个历史属性，
而 pinned vllm 版本的 tokenizer 缓存代码仍在读它。两者只在"是否可能含
`AddedToken` 对象"上有差异，对 vLLM 这里的用法（只是快照 special tokens）无影响。

**修复**：monkeypatch 一个 fallback 到 `all_special_tokens` 的 property。因为
vLLM worker 子进程是 spawn 出来的全新解释器（不会继承主进程内存里的
monkeypatch，但会继承 `PYTHONPATH`），补丁同时写在
`vllm_runner.py::load()`（主进程）和 `sitecustomize.py`（Python 解释器启动时
自动 import，覆盖 worker 子进程）两处。

**涉及文件**：`sitecustomize.py`、`unify_omnibench/models/vllm_backend/vllm_runner.py`

---

## 8. vLLM 多模态输入：单元素要不要用列表包装

**现象**：怀疑过 `qwen_omni_utils.process_mm_info()` 返回的单元素列表
（如 `videos=[tensor]`）直接传给 vLLM 的 `multi_modal_data` 会不会引发前面第1条
提到的 `NoneType` 类问题。

**结论**：vLLM 官方多模态示例对单条数据传**裸对象**、不用列表包装；传列表会让
vLLM 走"多条目"的 content-hash/`multi_modal_uuids` 记账路径。本仓库每条样本
最多一个 video/audio/image，所以在 `_build_one()` 里统一解包成裸对象
（`videos[0] if len(videos) == 1 else videos`），跟官方用法保持一致。

**涉及文件**：`unify_omnibench/models/vllm_backend/vllm_runner.py::_build_one`

---

## 9. `Qwen2VLImageProcessor` fast/slow 两种实现输出有细微差异

**现象**：同一张图/帧，不同后端（transformer / vLLM 服务端）处理器初始化方式
不同（有的默认 fast、有的默认 slow）会产出细微不同的像素预处理结果。

**修复**：所有本地后端统一显式传 `use_fast=True`（`Qwen2_5OmniProcessor.from_pretrained`），
避免三条路径各自依赖所在环境 transformers 版本的隐式默认值。

**涉及文件**：`unify_omnibench/models/local/qwen25omni.py`、
`unify_omnibench/models/vllm_backend/vllm_runner.py`

---

## 10. 死代码与过时测试：三个未注册的 Dataset Adapter + 两个跟着一起坏的测试

**现象**：`unify_omnibench/datasets/{daily_omni,omnibench,omnivideobench}.py`
三个文件各自定义了一个 Adapter 类，但 `@register_dataset(...)` 装饰器早就被
注释掉了（真正生效的是 `unified.py::UnifiedAdapter`，同时注册了这四个数据集名）。
这三个类完全不可达，但 `tests/test_adapters.py`/`test_runner_smoke.py`
仍然在直接构造它们期望的旧 schema（`qa_file`/`video_base_dir`/`Question`/
`Choice`），而实际注册的 `UnifiedAdapter` 需要的是 `data_file`+`media_root`+
统一 JSON schema——**这两个测试文件早就是在测一条不存在的代码路径，跑起来
会直接 `KeyError`/断言失败**。

**修复**：删除三个死 Adapter 文件 + `datasets/__init__.py` 里对应的 import；
重写 `test_adapters.py`/`test_runner_smoke.py`，改成针对 `UnifiedAdapter`
真实 schema 构造 fixture。

**教训**：跑一次 `pytest tests/` 是最快发现"文档/测试与实现脱节"的方式，本次
清理是靠人工审查发现的，说明 CI 里跑测试的习惯不能省。

**涉及文件**：`unify_omnibench/datasets/`、`tests/test_adapters.py`、
`tests/test_runner_smoke.py`

---

## 快速参考：升级 vllm/transformers 前该检查什么

1. `tests/test_qwen_omni_vllm.py` 跑一遍（覆盖 独立音频/交织音频 × 单条/batch调用 四种场景）
2. 确认 `mm_processor_cache_gb: 0` 是否还需要（缓存 None 占位符的 bug 是否已被上游修复）
3. 确认 `all_special_tokens_extended` 兼容性 shim 是否还需要（transformers/vllm 版本变了可能不需要了，也可能需要新的 shim）
4. `pytest tests/ -v` 跑一遍，确认没有引入新的 broken test

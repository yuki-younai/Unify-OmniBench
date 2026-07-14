# Agent ReAct 模式设计与实现

本文档介绍 Unify-OmniBench 中 Agent ReAct 评测模式的完整工作流、架构设计及各模块细节，帮助理解模型如何通过多轮“观察→思考→行动”循环来完成视频理解多选题（MCQ）。

---

## 1. 概述

ReAct（Reasoning + Acting）模式下，模型不再一次性读取视频后直接输出答案（direct 模式），而是扮演一个 **Agent**，在有限的步数预算内主动调用工具探索视频内容：

- **`get_frames`**：提取指定时间区间内的等距帧图片（视觉观察）
- **`get_audio`**：提取指定时间区间的音频片段（听觉观察）
- **`get_clip`**：提取指定时间区间的短视频片段（具备原始音轨的多模态观察）
- **`answer`**：提交最终答案，结束该样本的推理循环

每一轮 Agent 输出一个结构化 JSON，包含 `observation`（当前发现的总结）、`think`（推理与下一步决策）、`action`（工具调用或答案提交）。评测器执行工具、将结果反馈给模型，直至模型调用 `answer` 或达到最大步数上限。

---

## 2. 启动与配置

### 2.1 入口脚本 `eval_react.sh`

```bash
bash eval_react.sh
```

关键变量（默认值）：

| 变量 | 值 | 说明 |
|---|---|---|
| `BACKEND` | `vllm` | 推理后端（也支持 `transformer`） |
| `DATASETS` | `(daily_omni omnibench)` | 待评测数据集列表 |
| `RUN_MODE` | `react` | 固定位，传给 `run.py` |
| `MAX_NEW_TOKENS` | `4096` | 生成 token 上限（react 需要更多预算） |
| `TEMPERATURE` | `0.0` | 生成温度 |
| `GPUS_PER_WORKER` | `1` | 每个 worker 使用的 GPU 数 |
| `BYPASS_DURATION_CHECK` | `True` | 放宽 clip/audio 时长校验容差 |

多 worker 分片逻辑与 `eval.sh` 完全一致：按 `md5(uid) % num_shards` 分配样本。

### 2.2 Agent 配置文件 `config/agent.yaml`

```yaml
default:
  max_steps: 32          # 最大探索步数
  max_frames_len: 60     # get_frames 单次最多提取帧数
  max_audio_len: 300     # get_audio 单次最大秒数
  max_clip_len: 60       # get_clip 单次最大秒数
  dynamic_step: true     # 根据视频时长动态调整 max_steps
  generation:
    max_new_tokens: 4096
    temperature: 1.0
```

按 benchmark 可单独覆盖（`benchmarks:` 下的同名 key）。

---

## 3. 整体流程图

```
┌───────────────────────────────────────────────────────────────────┐
│  eval_react.sh                                                    │
│  ├─ 检查已完成 benchmark（check_completed.py）                      │
│  ├─ 多 worker 启动 run.py（GPU 分片）                              │
│  ├─ 等待所有 worker 完成                                          │
│  ├─ merge_shards.py 合并分片结果                                  │
│  └─ aggregate_results.py 生成 summary.md                          │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  run.py                                                           │
│  ├─ 读取 config/agent.yaml → agent_cfg                            │
│  ├─ 根据 benchmark 合并默认+特定配置                                │
│  ├─ 动态调整 limit_mm_per_prompt（image 上限 = max_frames_len）     │
│  ├─ 设置 run_dir（追加 _react 后缀）                               │
│  └─ 创建 ReActEvaluator 并调用 .run()                             │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  ReActEvaluator.run()                                            │
│  ├─ model.load()                                                 │
│  ├─ 断点续跑：_load_resume_items() → _scan_items()                │
│  ├─ 分片过滤、task_type 过滤、--limit 截断                       │
│  ├─ 逐样本 _evaluate_one()（带 ProgressManager 进度条）           │
│  ├─ _persist() 写入 + save_trajectory() 轨迹保存                  │
│  ├─ model.close()                                                │
│  ├─ 二次 _compact_items() 去重                                   │
│  └─ write_summary() 生成统计                                     │
└───────────────────────────────────────────────────────────────────┘
```

---

## 4. 单样本评测循环 `_evaluate_one()`

核心多轮 Agent 循环的逐步流程：

### 4.1 初始化

```
1. 从 Sample.media 中查找视频文件（_find_video）
2. 创建 VideoEnv(video_path)：
   - ffprobe 探测 duration / fps / has_audio
   - 创建临时目录存储工具输出的媒体文件
   - 读取环境变量 BYPASS_DURATION_CHECK 设置时长校验容差
3. 动态步数计算（dynamic_step）：
   max_steps = min(cfg_max_steps, 5 + int(duration / max_clip_len))
   - 更长视频分配更多探索步数
4. 构建初始 messages：
   [system]  → build_system_prompt()
   [user]    → "Video META: ... \n Question: ... \n Options: ..."
   [user]    → [NOTICE] 步数提示
```

### 4.2 逐轮循环（最多 max_steps 轮）

```
for step in 1 .. max_steps:
  ┌─ 1) 调用模型 _call_model(messages)
  │     └─ 将 messages 直传给 vLLM backend（不重新 build_messages）
  │        vLLM 的 process_mm_info 自动提取内嵌图片/音频/视频
  │
  ├─ 2) 解析输出 parse_action_json(raw)
  │     ├─ 去除 ```json``` fences
  │     ├─ json.loads 提取 {observation, think, confidence, action}
  │     ├─ action 字段归一化（字符串→dict）
  │     └─ 防御性类型检查（confidence float, observation/think str）
  │
  ├─ 3) 若 action_type == "answer" → 评测结束
  │     └─ 对比 answer.content vs sample.answer → is_correct
  │
  ├─ 4) 工具预验证 validate_action(action_type, action)
  │     ├─ get_frames/audio/clip: start/end 必须是数值
  │     ├─ get_frames: num 必须是 int
  │     └─ answer: content 必须是 str
  │     └─ 验证不通过 → 发送 [ERROR] INVALID_ACTION 消息，继续循环
  │
  ├─ 5) 执行工具 tool.execute(action, env)
  │     ├─ get_frames → env.get_frames(start, end, num)
  │     │   └─ ffmpeg 逐帧提取 jpg 图片
  │     ├─ get_audio → env.get_audio(start, end)
  │     │   ├─ NO_AUDIO 检查：!has_audio → 直接抛 ValueError
  │     │   ├─ ffmpeg 提取 wav
  │     │   └─ ffprobe 校验实际时长（dur_tol*5 容差）
  │     └─ get_clip → env.get_clip(start, end)
  │         ├─ ffmpeg copy 方式提取 mp4
  │         └─ ffprobe 校验视频/音频时长（多级校验）
  │     └─ 工具抛异常 → 捕获后发送 [ERROR] 消息，继续循环
  │
  ├─ 6) 追加消息到 messages：
  │     [assistant] → raw（模型原始输出 JSON）
  │     [user]      → 工具结果文本 + 内嵌媒体文件路径
  │
  ├─ 7) 记忆合并 _consolidate_memory(messages)
  │     └─ 旧轮次 assistant 消息 → 文本占位符
  │     └─ 旧轮次 user 消息图片/音频/视频 → 保留原 header 文本 + timestamps + 占位符
  │     └─ 仅保留最新 user 消息的媒体块（防 OOM / 不超出 limit_mm_per_prompt）
  │
  └─ 8) 步数提示 _append_step_notice(messages)
        └─ 剩余步数 ≤ 1 → "[NOTICE] FINAL STEP! You MUST provide your answer now."
        └─ 否则 → "[NOTICE] Step X/Y. Z steps remaining."
```

### 4.3 超时/错误处理

```
- 达到 max_steps 且未调用 answer → _error_result("max_steps reached without answer")
- 工具执行异常 → 追加 [ERROR] 文本消息，不影响其他步
- 视频缺失 → 直接返回 _error_result("no video in sample media")
- 模型输出不可解析（action_type="unknown"）→ parse_action_json 返回 unknown，
  ToolRegistry.get("unknown") 返回 None，发送 [ERROR] UNKNOWN_TOOL
```

---

## 5. 模块详解

### 5.1 `prompt.py` — System & User Prompt

**System Prompt 结构**（与 OmniAgent `video_env.py` 对齐，缩略为 MCQ-only）：

```
GLOBAL OPERATING RULES      — META 校验、音频约束、媒体持久化、策略效率等
STRATEGIC INSPECTION GUIDELINES — 视觉搜索、时间二分、音频转录、多模态动作分析
ACTIONS                     — 从 ToolRegistry 动态生成工具列表
STRICT EXECUTION PROTOCOL   — 证据溯源、置信度、截止管理
OUTPUT SCHEMA               — JSON 格式说明
```

**User Prompt 结构**（首条消息）：

```
Video META:
- duration_seconds: 120.00   ← 向下截断不四舍五入
- fps: 25.00
- has_audio: True

Question: What color is the car?
Options:
A. Red
B. Blue
C. Green
D. Yellow
When answering, set action.content to ONE uppercase letter (A, B, C, D).
```

### 5.2 `tools.py` — VideoEnv 与内置工具

**VideoEnv（视频环境）**：

| 属性/方法 | 说明 |
|---|---|
| `video_path` | 视频文件路径 |
| `tmp_dir` | 临时目录（`agent_xxx`），存储工具输出的 jpg/wav/mp4 |
| `dur_tol` | 时长校验容差（BYPASS_DURATION_CHECK=true→99999，否则 1.0s） |
| `_probe()` | ffprobe 探测 duration / fps / has_audio |
| `_ffmpeg()` | 执行 ffmpeg，cwd=tmp_dir，按 step_counter 编号 |
| `get_media_durations()` | ffprobe JSON 解析视频流/音频流实际时长 |
| `get_frames(start, end, num)` | 等距帧提取 → 返回 `List[MediaRef("image")]` |
| `get_audio(start, end)` | 音频提取 + 时长校验（dur_tol*5）→ `MediaRef("audio")` |
| `get_clip(start, end)` | 视频切片 + 多级校验（视频/音视频同步）→ `MediaRef("video")` |

**四大内置工具**（注册在 `ToolRegistry`）：

| 工具 | schema | 关键校验 |
|---|---|---|
| `get_frames` | `{"type":"get_frames","start":0.0,"end":10.0,"num":8}` | num ≤ max_frames_len |
| `get_audio` | `{"type":"get_audio","start":0.0,"end":15.0}` | !has_audio→ValueError; end ≤ start+max_audio_len; 时长校验 |
| `get_clip` | `{"type":"get_clip","start":5.0,"end":15.0}` | end ≤ start+max_clip_len; 时长+音视频同步校验 |
| `answer` | `{"type":"answer","content":"A"}` | content 必为字符串 |

> **扩展性**：新增工具只需继承 `BaseTool` 并调用 `ToolRegistry.register()`，prompt 会自动包含新工具的 schema。

### 5.3 `action_parser.py` — 模型输出解析

**`parse_action_json(raw)`** 处理流程：

1. 检测并剥离 ````json``` 代码块标记
2. 正则提取最外层 `{...}` JSON
3. `json.loads` 解析
4. 防御性归一化：
   - `action` 为字符串 → `{"type": action}`
   - `action` 非 dict → `{}`
   - `confidence` 无法转为 float → `0.0`
   - `observation`/`think` 非 str → 空串

**`validate_action(action_type, action)`** 执行前字段校验：

- `get_frames`：`start`/`end` 必须为数值，`num` 必须为 int
- `get_audio`/`get_clip`：`start`/`end` 必须为数值
- `answer`：`content` 必须为 str

### 5.4 `react_evaluator.py` — Agent 调度核心

**`ReActEvaluator`** 负责完整的评测生命周期：

```
run()
├─ model.load()
├─ _compact_items()          # 去除上一轮重跑的重复记录
├─ _load_resume_items()      # 断点续跑：加载 items.jsonl
├─ _scan_items()             # 区分 done/retry uid
├─ 分片过滤 / task_type 过滤 / --limit 截断
├─ 进度条 ProgressManager
├─ for each pending sample:
│   └─ _evaluate_one()       # 单样本多轮 Agent 循环
│       └─ _persist()        # 写入 items.jsonl + failed.jsonl + 轨迹保存
├─ model.close()
├─ _compact_items()          # 二次去重
└─ write_summary()           # 统计准确率/breakdown
```

**`_consolidate_memory()` — 记忆合并机制**（对齐 OmniAgent 的 `_replace_old_media`）：

| 步骤 | 操作 |
|---|---|
| assistant 消息（非最后轮） | `content` 替换为 `[{"type":"text","text":"[MEDIA OMITTED - Refer to your Observation]"}]` |
| user 消息（非最后轮，含媒体） | 保留 header 文本 + timestamps 列表 → `"Frames 10.00s-20.00s Timestamps: [10.00s, 12.50s, ...] [MEDIA OMITTED ...]"` |
| 最新 user 消息 | **保留完整媒体块**（图片/音频/视频路径） |

> 此举确保长视频的多轮探索不会累积无限图片导致 OOM 或超出 vLLM 的 `limit_mm_per_prompt` 上限。

---

## 6. 与 Direct 模式的差异对齐

| 维度 | Direct 模式 | ReAct 模式 |
|---|---|---|
| **推理后端** | `build_messages()` 从 Sample 构建一次性对话 | `req.messages` 直传多轮已构建的 messages |
| **vLLM limit_mm_per_prompt** | `{"image":1,"video":1,"audio":1}` | `{"image":max_frames_len,"video":1,"audio":1}` |
| **结果目录** | `results/<ds>/<model>_<backend>_norm/` | `results/<ds>/<model>_<backend>_norm_react/` |
| **MAX_NEW_TOKENS** | 2048 | 4096 |
| **system prompt** | 无（仅 user） | 完整的 Agent 行为准则（含工具列表） |
| **温度** | `0.0` | `1.0`（促进探索多样性） |

> 断点续跑、分片、merge、summary 统计等运维逻辑与 direct 模式完全一致。

---

## 7. 轨迹保存（Trajectory）

每个样本的完整推理过程会保存为两种格式：

### 7.1 OpenAI Messages 格式 JSON

路径：`trajectories/<uid>.json`

```json
[
  {"role": "system", "content": [{"type": "text", "text": "..."}]},
  {"role": "user", "content": [{"type": "text", "text": "Video META: ..."}]},
  {"role": "user", "content": [{"type": "text", "text": "[NOTICE] Step 0/8..."}]},
  {"role": "assistant", "content": "{\"observation\":\"...\",\"action\":{...}}"},
  ...
]
```

> 媒体路径会被 `_sanitize_messages()` 替换为 basename，避免泄露绝对路径。

### 7.2 HTML 可视化

路径：`trajectories/<uid>.html`

一个自包含的 HTML 页面，呈现：
- 题目、选项、正确答案/模型答案标记
- 每步的 raw JSON 输出（可折叠查看）
- 错误步的警告标识
- 最终结果 badge（Correct/Incorrect/ERROR）

**注意**：对话紧凑化（`_consolidate_memory`）后的中轮消息会被标记为 `[MEDIA OMITTED]`，轨迹快照在 **answer 之前** 或 **max_steps 耗尽时** 截取，因此 HTML 中可能看到已被合并的消息。完整原始的媒体路径记录在 `items.jsonl` 的 `meta.history` 中。

---

## 8. 环境变量一览

| 环境变量 | 作用 | 默认值 |
|---|---|---|
| `BYPASS_DURATION_CHECK` | 放宽 clip/audio ffprobe 时长校验容差 | `false`（eval_react.sh 设为 `True`） |
| `MAX_STEPS_OVERRIDE` | 覆盖 agent.yaml 的 max_steps | 无 |
| `VLLM_WORKER_MULTIPROC_METHOD` | vLLM worker 多进程模式 | `spawn` |
| `CUDA_VISIBLE_DEVICES` | 可用 GPU 列表 | `0,1,2,3,4,5,6,7` |

---

## 9. 统计指标（Summary）

`write_summary()` 生成的统计中，ReAct 模式额外包含：

- **`meta.steps`**：每个样本实际使用的探索步数
- **`meta.agent_confidence`**：Agent 提交 answer 时的置信度
- **`meta.duration`**：视频时长
- **breakdown 维度**：除 `task_type` 外，还可按步数区间、置信度区间统计

---

## 10. 关键设计决策

1. **OmniAgent 完全对齐**：prompt 结构、工具校验（`validate_action`）、时长校验（`BYPASS_DURATION_CHECK` + `dur_tol`）、记忆合并（`_consolidate_memory`）、步数提示（`[NOTICE]`）、NO_AUDIO 保护、MediaRef 规范化——所有细节均参考 `OmniAgent/agent_system/environments/env_package/video_env.py` 实现。

2. **防御性解析**：模型输出可能格式不规范（裸字符串、非 dict action、丢失字段），`parse_action_json` 做了五层兜底。

3. **工具独立性**：VideoEnv 作为纯粹的视频环境，工具通过 ToolRegistry 插件式注册，新增工具无需修改 evaluator 或 prompt 模板。

4. **运维健壮性**：断点续跑、分片合并、去重 compaction、失败自动重试、临时文件清理——这些与 direct 模式共享同一套基础设施。

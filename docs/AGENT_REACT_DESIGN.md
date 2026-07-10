# Agent ReAct 测评迁移方案

> 将 OmniAgent 的 OTA（Observation-Think-Action）Agent 循环迁移到 Unify-OmniBench，
> 作为现有简单输入-输出测评的增强模式，后端复用现有 vllm/transformer/openai。

---

## 1. 核心差异对比

| | 现有 Unify-OmniBench | OmniAgent ReAct |
|---|---|---|
| 推理模式 | 单次 `generate()` | 多轮迭代，每轮输出 JSON action |
| 媒体输入 | 一次喂入全部视频+音频 | 按需请求片段（ffmpeg 实时裁剪） |
| 上下文管理 | 固定 `system + user` | 多轮累积 + `[MEDIA OMITTED]` 记忆巩固 |
| 工具调用 | 无 | 3 工具（get_frames / get_audio / get_clip）+ answer |
| 并发 | batch / thread | 每 GPU 一个独立进程，JoinableQueue |
| Batch 推理 | `generate_batch()` | 无 batch，逐条串行 |

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                      eval.sh                             │
│  (已有，多 worker 分布式启动不变)                          │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│  run.py / Runner (现有)                                   │
│    └── 新增: run_mode = "direct" | "react"               │
└──────────┬──────────────────────────────────────────────┘
           │
           ├── run_mode=direct (现有逻辑，不变)
           │
           └── run_mode=react (新增)
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│  ReActEvaluator (新模块: unify_omnibench/agent/)         │
│                                                          │
│  for sample in pending_samples:                           │
│      env = VideoEnv(sample.media[0].path)                │
│      messages = build_initial_prompt(sample, env.meta)   │
│                                                          │
│      for step in 1..max_steps:                           │
│          raw = model.generate(messages)  ← 复用现有后端   │
│          parsed = parse_action_json(raw)                  │
│          if parsed.action == "answer":                    │
│              end                                          │
│          obs = env.execute(parsed.action)  ← 工具执行     │
│          messages = append_turn(messages, raw, obs)       │
│                                                          │
│      persist_result(sample, steps, reward, history)      │
└──────────────────────────────────────────────────────────┘
```

**关键设计原则**：
- **后端完全复用**：`model.generate(req)` 的调用方式和现有 Runner 一致，不做任何修改
- **Runner 新增模式开关**：`run_mode=direct` 走老逻辑，`run_mode=react` 走 Agent 循环
- **工具系统独立**：放在 `unify_omnibench/agent/tools.py`，不污染现有代码

---

## 3. 新增模块设计

### 3.1 目录结构

```
unify_omnibench/
├── agent/                      # ★ 新增
│   ├── __init__.py
│   ├── react_evaluator.py      # 核心 Agent 循环
│   ├── tools.py                # 工具实现（get_frames, get_audio, get_clip）
│   ├── action_parser.py        # JSON 解析 + action 验证
│   ├── video_env.py            # 视频环境（ffprobe, ffmpeg）
│   └── prompt.py               # ReAct system prompt 模板
├── runner.py                   # 修改：新增 run_mode 分支
├── core/types.py               # 修改：新增 ReactResult 类型
├── config/
│   └── dataset_config.yaml     # 修改：新增 react 相关配置
```

### 3.2 核心循环 — `react_evaluator.py`

```python
class ReActEvaluator:
    def __init__(self, dataset, model, cfg):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg
        self.max_steps = cfg.get("react", {}).get("max_steps", 32)
        self.tools = ToolRegistry()   # 注册 get_frames / get_audio / get_clip

    def run(self) -> Dict[str, Any]:
        self.model.load()
        all_samples = list(self.dataset)
        done_uids = self._load_done_uids()  # 复用 Runner 的断点续跑逻辑

        for sample in all_samples:
            if sample.uid in done_uids:
                continue
            result = self._evaluate_one(sample)
            self._persist(result)

        return write_summary(...)

    def _evaluate_one(self, sample) -> ReActResult:
        video_path = self._find_video(sample.media)  # 取第一个 video MediaRef
        env = VideoEnv(video_path)

        # 构建初始 prompt（含视频元数据和题目）
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(sample, env.meta())
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        history = []
        total_reward = 0.0
        for step in range(1, self.max_steps + 1):
            # 1. 调用模型（复用现有后端！）
            raw = self.model.generate(InferenceRequest(
                sample=self._make_chat_sample(messages),  # 把 messages 包装为 Sample
                modality_mode="text",                     # media 已内嵌在 messages 中
                prompt_template=None,
                generation_kwargs={"max_new_tokens": 2048},
            ))

            # 2. 解析 JSON action
            parsed = parse_action_json(raw)
            history.append({"step": step, "raw": raw, "parsed": parsed})

            # 3. 执行工具 / 提交答案
            if parsed.action_type == "answer":
                reward = env.score_answer(sample, parsed.content)
                total_reward = reward
                break

            obs_text = self.tools.execute(parsed.action, env)

            # 4. 记忆巩固：追加 assistant + user(observation) 到 messages
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": obs_text})

            # 5. 替换旧媒体为占位符（节省上下文）
            self._consolidate_memory(messages)

        return ReActResult(
            uid=sample.uid, dataset=sample.dataset,
            steps=len(history), reward=total_reward,
            answer=parsed.content if parsed.action_type == "answer" else None,
            history=history,
        )
```

### 3.3 工具系统 — `tools.py`

```python
class VideoEnv:
    """视频环境：ffprobe 探测 + ffmpeg 裁剪 + 执行 action"""
    def __init__(self, video_path):
        self.video_path = video_path
        self.meta = self._probe()        # duration, fps, has_audio
        self.tmp_dir = tempfile.mkdtemp()
        self.step_counter = 0

    def meta(self) -> dict:
        return self.meta

    def execute(self, action: dict) -> tuple[str, list[MediaRef]]:
        """执行 action，返回 (observation_text, new_media_refs)"""
        self.step_counter += 1
        typ = action["type"]
        if typ == "get_frames":
            frames = self._get_frames(action["start"], action["end"], action.get("num", 10))
            return f"[Frames {action['start']:.1f}s-{action['end']:.1f}s (num={len(frames)})]", frames
        elif typ == "get_audio":
            audio = self._get_audio(action["start"], action["end"])
            return f"[Audio {action['start']:.1f}s-{action['end']:.1f}s]", [audio]
        elif typ == "get_clip":
            clip = self._get_clip(action["start"], action["end"])
            return f"[Clip {action['start']:.1f}s-{action['end']:.1f}s]", [clip]

    # 底层用 subprocess 调 ffmpeg（与 OmniAgent 完全一致）
    def _get_frames(self, start, end, num): ...
    def _get_audio(self, start, end): ...
    def _get_clip(self, start, end): ...
```

### 3.4 Prompt 模板 — `prompt.py`

```python
SYSTEM_PROMPT = """You are an AI agent with visual and audio perception capabilities.
Your task is to answer a question about a video by actively exploring its content.

Available tools (respond with exactly one JSON per turn):
- {"type": "get_frames", "start": 0.0, "end": 10.0, "num": 8}
- {"type": "get_audio", "start": 0.0, "end": 15.0}
- {"type": "get_clip",  "start": 5.0, "end": 15.0}
- {"type": "answer", "content": "A"}  ← submit final answer

Response format (MUST be valid single-line JSON):
{
  "observation": "<summary of what you just perceived, with evidence tags>",
  "think": "<reasoning: evidence review → gap analysis → deduction → next action>",
  "confidence": <0.0-1.0>,
  "action": <one of the tool types above>
}

Rules:
- You CANNOT answer on the first step (must explore at least once).
- Use [Frames Xs-Ys] / [Audio Xs-Ys] / [Clip Xs-Ys] tags to cite evidence.
- Confidence < 0.9 → you likely need more evidence.
- Previously viewed media is marked as [MEDIA OMITTED] to save context;
  rely on your own observation summaries."""

USER_TEMPLATE = """Video: duration={duration:.1f}s, fps={fps:.1f}, has_audio={has_audio}

Question: {question}
Options: {choices}
(answer format: single capital letter A/B/C/D)"""
```

---

## 4. 与现有系统的集成

### 4.1 `runner.py` 改动（最小化）

```python
class Runner:
    def run(self) -> Dict[str, Any]:
        if self.cfg.get("run_mode") == "react":
            return self._run_react()
        return self._run_direct()   # 现有逻辑不变

    def _run_react(self):
        from .agent.react_evaluator import ReActEvaluator
        evaluator = ReActEvaluator(self.dataset, self.model, self.cfg)
        return evaluator.run()

    def _run_direct(self):
        ...  # 现有 run() 逻辑，完全不动
```

### 4.2 `run.py` 改动

```python
p.add_argument("--run-mode", default="direct", choices=("direct", "react"))
```

### 4.3 `dataset_config.yaml` 改动

```yaml
react:
  max_steps: 32
  system_prompt: |
    You are an AI agent ...  # ← 默认 ReAct system prompt
```

### 4.4 后端适配

**关键：后端代码完全不需要改**。Agent 循环中每一轮都调用 `model.generate(req)`，和现有逻辑一致。唯一的区别是 `modality_mode` 可能从 `"av"` 变为 `"text"`（当媒体已内嵌在 `messages` 的 content 中时）。

对于 `openai_chat` 后端：`build_messages()` 需要新增一个模式——当 `req` 中已经包含预构建的 `messages` 时，直接透传而不重新拼装。加一个 `req.prefab_messages: Optional[List[Dict]]` 字段即可。

### 4.5 断点续跑

完全复用现有逻辑：`_load_done_uids()` 从 `items.jsonl` 读已完成样本 uid，已成功的跳过。Agent 中间步的失败不计入 done（只计入 done 的条件是最终 submit answer 且无 error）。

---

## 5. 实施步骤

| 阶段 | 改动 | 风险 |
|---|---|---|
| **Phase 1** | 新建 `unify_omnibench/agent/` 模块，实现 `VideoEnv` + 工具 + `ReActEvaluator` | 纯新增模块，不影响现有功能 |
| **Phase 2** | `runner.py` 加 `run_mode` 分支，`run.py` 加 `--run-mode` 参数 | 改动 10 行，`direct` 默认路径不变 |
| **Phase 3** | `InferenceRequest` 加 `prefab_messages` 字段，`openai_chat.py` 适配 | 向后兼容，默认 None |
| **Phase 4** | `eval.sh` 支持 `RUN_MODE=react`，`aggregate_results.py` 支持 Agent 指标 | 纯新增 |

---

## 6. 复用与不迁移的内容

| 复用（直接搬） | 不迁移 |
|---|---|
| ✅ OTA 循环逻辑 + 4 工具 | ❌ `multiprocessing` 生产者-消费者（我们已有 `eval.sh` 的多 worker 分片） |
| ✅ JSON action 解析 | ❌ 视频预分段 + chunk 调度（OmniAgent 的特化逻辑，当前不需要） |
| ✅ ffmpeg 视频处理命令 | ❌ `ModelManager`（我们用现有的 backend 体系） |
| ✅ system prompt 模板 | ❌ DashScope LLM-as-Judge（FF 题型暂不支持，先用 MCQ 精确匹配） |
| ✅ 记忆巩固机制 | ❌ Trajectory Collector（RL 训练相关，评测不需要） |
| ✅ 自适应步数 | |

---

## 7. 扩展性设计

### 7.1 工具插件注册机制

新增工具无需修改 `ReActEvaluator`，只需实现 `BaseTool` 并注册：

```python
# unify_omnibench/agent/tools.py

class BaseTool(ABC):
    """工具基类——新增工具只需继承并实现 execute + schema"""
    name: str                          # "get_frames"
    description: str                   # system prompt 中的工具描述
    json_schema: Dict[str, Any]         # action 的 JSON schema（用于 prompt 生成 + 校验）

    @abstractmethod
    def execute(self, action: dict, env: "VideoEnv") -> ToolResult:
        """执行工具，返回 (observation_text, media_refs, metadata)"""

class ToolRegistry:
    """插件式工具注册表"""
    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool):
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> BaseTool:
        return cls._tools[name]

    @classmethod
    def build_system_prompt(cls) -> str:
        """自动生成 tools 描述块，注入 system prompt"""
        return "\n".join(f"- {t.json_schema}" for t in cls._tools.values())

# ── 内置工具 ──
ToolRegistry.register(GetFramesTool())
ToolRegistry.register(GetAudioTool())
ToolRegistry.register(GetClipTool())
ToolRegistry.register(AnswerTool())

# ── 扩展示例：新增一个字幕提取工具 ──
class GetSubtitlesTool(BaseTool):
    name = "get_subtitles"
    description = "Extract embedded subtitles from a time range"
    json_schema = {"type": "get_subtitles", "start": 0.0, "end": 30.0}

    def execute(self, action, env):
        text = env._extract_subtitles(action["start"], action["end"])
        return ToolResult(f"[Subtitles {action['start']}s-{action['end']}s]: {text}", [])

ToolRegistry.register(GetSubtitlesTool())   # ← 一行注册，system prompt 自动更新
```

### 7.2 Loop 策略接口

Agent 循环逻辑可插拔，支持不同策略（标准 ReAct / Tree-of-Thought / 自定义）：

```python
# unify_omnibench/agent/loop_strategies.py

class LoopStrategy(ABC):
    """Agent 循环策略——新增循环逻辑只需实现此接口"""

    @abstractmethod
    def run(self, sample, model, env, tools: ToolRegistry, cfg: dict) -> ReActResult:
        """执行一次完整的 Agent 循环，返回结果"""


class ReActLoop(LoopStrategy):
    """标准 ReAct 循环：逐轮调用模型→解析 action→执行工具→追加上下文"""
    def run(self, sample, model, env, tools, cfg):
        messages = self._build_init_messages(sample, env, tools)
        for step in range(cfg.get("max_steps", 32)):
            raw = self._generate(model, messages, cfg)
            parsed = parse_action_json(raw)
            if parsed.action_type == "answer":
                return self._finalize(sample, parsed, step, messages)
            result = tools.get(parsed.action_type).execute(parsed.action, env)
            self._append_turn(messages, raw, result)
            self._consolidate_memory(messages)
        return self._timeout_result(sample, messages)   # 超步数未作答


class TreeOfThoughtLoop(LoopStrategy):
    """ToT 循环：每步生成 3 个候选 action，选择最优路径"""
    def run(self, sample, model, env, tools, cfg): ...


class ReActEvaluator:
    def __init__(self, dataset, model, cfg):
        ...
        strategy_name = cfg.get("react", {}).get("loop_strategy", "react")
        self.loop = LOOP_REGISTRY[strategy_name]()   # ← 按配置选择策略
```

### 7.3 Action 解析器插件

不同 prompt 模板可能输出不同 JSON 格式，解析器也可插拔：

```python
class ActionParser(ABC):
    @abstractmethod
    def parse(self, raw: str) -> ParsedAction: ...

class DefaultOTAParser(ActionParser):
    """解析 OmniAgent 标准 OTA JSON"""
    def parse(self, raw): ...

class CustomParser(ActionParser):
    """自定义解析逻辑"""
    def parse(self, raw): ...
```

### 7.4 配置驱动

所有扩展点通过 `dataset_config.yaml` 控制，无需改代码：

```yaml
daily_omni:
  react:
    enabled: true
    loop_strategy: react          # react | tree_of_thought | custom
    max_steps: 32
    tools:                        # 启用哪些工具
      - get_frames
      - get_audio
      - get_clip
      - answer
    system_prompt: |
      You are an AI agent...
    generation:
      max_new_tokens: 2048
      temperature: 0.0
```

---

## 8. 风险与注意事项

1. **显存压力**：Agent 循环中每步都可能新增媒体（裁剪的视频片段），多轮后 GPU 显存可能不足。建议 React 模式降低 `gpu_memory_utilization` 或限制 `max_steps`。
2. **上下文爆炸**：当前 `max_model_len=32768`，多轮 + 内嵌媒体可能超限。记忆巩固（`[MEDIA OMITTED]`）可缓解，但需要实际验证。
3. **vLLM batch 模式**：Agent 循环是逐条串行的，无法利用 `generate_batch`。但可以在多 worker 层面并行（每个 worker 独立跑不同的样本）。
4. **openai 后端**：需要通过 `prefab_messages` 字段透传预构建的 conversation，避免 `build_messages()` 重新编码媒体。

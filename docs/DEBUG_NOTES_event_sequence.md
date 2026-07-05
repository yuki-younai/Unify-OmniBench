# Event Sequence 排查笔记（推测为主，未最终定论）

> 背景：Daily-Omni 评测中，openai backend（走 vllm-omni HTTP 服务）在 "Event Sequence"
> (纯视觉事件先后顺序题) 上明显落后 transformer backend（本地 Transformers 推理，作为
> 准确率参照基准）：约 **41.7% vs 55.6%**，而其他题型差距不大甚至已追平。以下是本轮排查
> 过程中的发现、已做的改动、以及仍然只是**推测、未被真实数据完全证实**的假设。

## 已确认排除的方向

1. **`use_audio_in_video` 音频跳过逻辑** —— 客户端正确跳过独立音频块、正确设置
   `mm_processor_kwargs.use_audio_in_video`。✅ wiring 正常。
2. **`second_per_grid_ts`（视频时序位置编码）本身** —— 显式设置
   `mm_processor_kwargs.fps` 后，合成视频（APPLE→BANANA→...→EGGPLANT，纯视觉）的首/末
   词、完整顺序三项测试全部通过，且真实数据准确率**分毫未变**（改前改后一致），说明这个
   参数从头到尾就没真的错，只是"侥幸对齐"变成了"显式对齐"，数值没变。
3. **音画交织（use_audio_in_video 的位置交织逻辑）本身把纯视觉顺序搞乱** —— 给合成视频
   用 `ffmpeg` 混流真实音轨后重跑，交织模式下顺序依旧全部答对。**这个假设被证伪**。
4. **视频时长导致的帧采样密度问题**（服务端 `--media-io-kwargs num_frames=128,fps=2`，
   猜测长视频会因 128 帧上限导致有效 fps 下降）—— 实测 30s vs 60s 视频上的差距几乎相等
   （13.0pt vs 13.2pt），跟时长无关。**这个假设也被证伪**（60s 视频 120 帧仍 <128 上限，
   本来就不该触发）。
5. **`video_url.num_frames`/`fps` 字段** —— 确认是死代码，vLLM 的 `VideoURL` TypedDict
   只认 `url` 字段，其余静默丢弃。真正生效的是服务端 `--media-io-kwargs` 启动参数 +
   `mm_processor_kwargs.fps`（显式传递，见上）。

## 当前最新假设（未证实，纯推测）

对比三条推理路径的代码后发现一个此前被忽略的分歧：

| Backend | `use_audio_in_video` | 视频位置编码方式 |
|---|---|---|
| **transformer**（准确率基准） | 硬编码 `False`（注释："匹配原始 Daily-Omni 评测方式"） | 视频独立编码、音频单独输入，**不交织** |
| vllm_runner.py（离线批量引擎） | `True` | 为绕开**离线引擎**"独立音频张量污染后续请求"的状态 bug |
| openai_chat.py（本次排查对象，HTTP 单请求） | 之前也是 `True`（照抄了 vllm_runner.py 的理由） | 交织 |

**推测**：openai 路径照抄了一个不属于它的理由（HTTP 单请求模式不共享离线引擎的状态，
理论上不会撞上那个 bug），导致它和"标准答案"transformer 用了不同的视频时序编码策略。
这**可能**就是 Event Sequence 差距长期存在、且和视频时长/前几轮时序修复都无关的真正原因
——但目前只有"证据不矛盾"（时长无关、修复无效都符合这个假设），**没有做真实数据的
对照实验验证**，不能算已证实。

## 已做的改动（实验性，待验证）

- `openai.yaml` / `openai_chat.py`：`use_audio_in_video` 默认改为 `False`，音频回退为
  独立发送（16kHz wav），对齐 transformer 的编码策略。
- `tests/test_qwen_omni_openai.py`：
  - 新增/修复 client-side wiring 检查（双向断言当前配置的实际行为）。
  - 新增 `check_temporal_ordering()`：合成视频（客观已知事件顺序）+ `ffmpeg` 混流真实
    音轨，快速验证时序编码链路，替代等 20 分钟的全量 eval。
  - 修了一个合成视频没有音轨导致 av 变体请求直接 500 的测试 bug。
- `run.py` / `runner.py`：加了 `--limit` / `--task-type` 参数，可只跑指定 task_type
  的小样本，不用等全量 1197 条。

## 待验证 / 下一步

```bash
python run.py --backend openai --dataset daily_omni --model-name Qwen2.5-Omni-7B \
    --task-type "Event Sequence" --limit 30
```

- 重点看这批 Event Sequence 准确率是否明显向 transformer 的 55.6% 靠近。
- 同时抽一批 "AV Event Alignment" 验证是否退步（该题型可能确实需要交织，改动可能是
  有取有舍，不一定是纯粹的净收益）。
- 如果这次改动依然无效，说明"编码策略不一致"这个假设也是错的，需要重新从
  prompt 模板差异 / 图像分辨率差异 / 采样温度等其他维度排查。

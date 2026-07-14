# FutureOmni 迁移方案

> 将 FutureOmni（全模态未来预测 benchmark）接入 Unify-OmniBench 统一评测框架。

---

## 1. 核心挑战：`split_point` 视频截断

FutureOmni 与现有 benchmark 的根本区别：

```
现有 benchmark:  看完整视频 → 回答问题
FutureOmni:      看视频前半段（到 split_point 帧） → 预测接下来发生什么
```

**解法**：转换阶段用 ffmpeg 预分割视频，截取 `[0, split_point]` 作为"前提视频"喂给模型。模型看不到 split_point 之后的内容。

---

## 2. 文件清单

| 文件 | 说明 |
|---|---|
| `script/convert_futureomni.py` | 新增：原始数据 → 统一 JSON + 预分割视频 |
| `unify_omnibench/config/dataset_config.yaml` | 修改：新增 future_omni entry |
| `unify_omnibench/config/agent.yaml` | 可选：FutureOmni 的 Agent 配置 |

---

## 3. 数据转换 — `script/convert_futureomni.py`

### 3.1 输入

```json
// 原始 FutureOmni annotation (来自 HuggingFace)
{
    "id": 0,
    "question": "Given the premise event: '...pausing on Four...', ...",
    "options": ["A. ...", "B. ...", "C. ...", "D. ...", "E. ..."],
    "answer": "B",
    "original_video": "uu8c_EH8VPE.mp4",
    "split_point": 227,
    "video_domain": "education",
    "audio_type": "Sound",
    "forecasting_pattern": "Routine Sequences"
}
```

### 3.2 输出

```json
// 统一 JSON 格式
{
    "id": "future_omni:0",
    "question": "Given the premise event: ...",
    "choices": ["A. ...", "B. ...", ...],
    "answer": "B",
    "video_path": "media/video/future_omni/0_premise.mp4",   // 截断后的视频
    "task_type": "Forecasting",
    "category": "education",
    "audio_type": "Sound",
    "meta": {
        "split_point": 227,
        "original_video": "uu8c_EH8VPE.mp4",
        "forecasting_pattern": "Routine Sequences"
    }
}
```

### 3.3 伪代码

```python
def convert(args):
    mm_root = os.path.join(args.out_dir, "media", "video", "future_omni")
    os.makedirs(mm_root, exist_ok=True)

    records = load_annotations(args.annotation_file)
    samples = []

    for r in records:
        src_video = os.path.join(args.video_dir, r["original_video"])

        # 探测视频 fps
        fps = probe_fps(src_video)

        # 计算截断时间点
        trunc_sec = r["split_point"] / fps

        # ffmpeg 截断：保留 [0, trunc_sec]
        dst_video = os.path.join(mm_root, f"{r['id']}_premise.mp4")
        if not os.path.exists(dst_video):
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", src_video, "-t", f"{trunc_sec:.3f}",
                "-c", "copy", dst_video,
            ], check=True)

        samples.append({
            "id": f"future_omni:{r['id']}",
            "question": r["question"],
            "choices": r["options"],
            "answer": r["answer"],
            "original_video": r["original_video"],
            "video_path": f"media/video/future_omni/{r['id']}_premise.mp4",
            "task_type": "Forecasting",
            "category": r.get("video_domain", ""),
            "audio_type": r.get("audio_type", ""),
            "meta": {
                "split_point": r["split_point"],
                "forecasting_pattern": r.get("forecasting_pattern", ""),
            },
        })

    # 写统一 JSON
    data_file = os.path.join(args.out_dir, "data", "future_omni.json")
    with open(data_file, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"FutureOmni: {len(samples)} samples → {data_file}")
    print(f"Videos:     {mm_root}/")
```

---

## 4. 配置 — `dataset_config.yaml`

```yaml
future_omni:
  data_file: data/future_omni.json
  modality: av
  use_audio_in_video: false        # 视频自带音频，独立音频无时间戳绑定
  system_prompt: "You are Qwen, ..."
  prompt_template: |
    Your task is to predict the most likely future event based on the {media_desc}.
    Select the single most likely answer from the given choices.
    Question: {question}
    Choices: {choices}
    Your answer should be a capital letter: A, B, C, D, or E. Don't generate any other text.
```

**为什么选 `use_audio_in_video: false`**：

FutureOmni 视频用 ffmpeg `-c copy` 截断（流拷贝），音视频轨保持原始同步。`use_audio_in_video=false` 表示音频从独立文件读取——而截断后的 `.mp4` 容器自带音轨，`process_mm_info` 设为 False 时会从容器中独立解码音频，时序自然对齐。

---

## 5. 评测命令

```bash
# 转换数据
python script/convert_futureomni.py \
    --annotation-file /path/to/futureomni_annotations.json \
    --video-dir /path/to/futureomni_videos \
    --out-dir /apdcephfs_hldy/share_304318596/weiyangguo/Datasets/Unify-OmniBench

# 单题测试
python tests/test_bench_show.py --bench future_omni --index 1

# 正式评测
BACKEND=vllm DATASETS=(future_omni) GPUS_PER_WORKER=0 bash eval.sh
```

---

## 6. 与现有 benchmark 的对比

| | daily_omni | omnivideobench | future_omni |
|---|---|---|---|
| 任务 | 音视频理解 | 视频理解 | **未来预测** |
| 输入 | 完整视频 | 完整视频 | **截断到 split_point** |
| 题型 | 4 选 1 | 4 选 1 | **5 选 1** |
| 核心能力 | 跨模态关联 | 视觉理解 | **因果推理 + 时序预测** |

---

## 7. Agent 模式（可选）

FutureOmni 天然适配 Agent 模式——"先看完前提片段 → 决定是否需要更多信息 → 预测未来"。只需在 `agent.yaml` 中加配置：

```yaml
future_omni:
  max_steps: 16       # 预测任务步数可以少一些
  dynamic_step: true
```

Agent 的优势：可以逐步探索截断点附近的帧和音频，做更精细的时序因果推理。

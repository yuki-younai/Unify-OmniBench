#!/usr/bin/env python3
"""一次性脚本：把本地 Unify-OmniBench 工作目录（omnibench + daily_omni）规范化并推送到
HuggingFace `main` 分支，重构为多 subset（config_name = omnibench / daily_omni）结构，
这样 Dataset Viewer 页面上会出现下拉框，可以按 bench 切换查看对应的 case。

请在能直接看到以下本地目录的机器/终端上运行本脚本（LOCAL_ROOT 按需修改）：
    /apdcephfs_hldy/share_304318596/weiyangguo/Datasets/OmniDatasets/Unify-OmniBench/
        data/omnibench.json
        data/daily_omni.json
        media/image/omnibench/*.png
        media/audio/omnibench/*.mp3
        media/video/daily_omni/*.mp4
        media/audio/daily_omni/*.wav

运行方式：
    pip install -U huggingface_hub pyarrow
    export HF_TOKEN=hf_xxx        # 需要对 yukiyounai/Unify-OmniBench 有写权限的 token
    python3 push_daily_omni_to_hf.py

可选参数（环境变量）：
    LOCAL_ROOT   本地数据根目录，默认见下方 DEFAULT_LOCAL_ROOT
    DRY_RUN=1    只在本地构建 parquet 并打印信息，不实际上传（用于先自检）
"""
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ID = "yukiyounai/Unify-OmniBench"
REF = "main"
DEFAULT_LOCAL_ROOT = (
    "/apdcephfs_hldy/share_304318596/weiyangguo/Datasets/OmniDatasets/Unify-OmniBench"
)
LOCAL_ROOT = os.environ.get("LOCAL_ROOT", DEFAULT_LOCAL_ROOT)
DRY_RUN = os.environ.get("DRY_RUN") == "1"

MEDIA_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def hf_uri(rel_path):
    if not rel_path:
        return None
    return f"hf://datasets/{REPO_ID}@{REF}/{rel_path}"


def build_omnibench_parquet(out_path):
    src = os.path.join(LOCAL_ROOT, "data", "omnibench.json")
    records = json.load(open(src, encoding="utf-8"))
    print(f"[omnibench] records: {len(records)} (from {src})")

    field_names = [
        "id", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer",
        "video_path", "audio_path", "image_path", "task_type", "category", "duration",
        "meta_audio_type", "meta_audio_content", "meta_image_content",
    ]
    cols = {k: [] for k in field_names}
    for r in records:
        for k in field_names:
            if k in ("audio_path", "image_path"):
                continue
            cols[k].append(r.get(k))
        cols["audio_path"].append({"bytes": None, "path": hf_uri(r.get("audio_path"))})
        cols["image_path"].append({"bytes": None, "path": hf_uri(r.get("image_path"))})

    arrays, names = [], []
    for name in ["id", "question", "choice_a", "choice_b", "choice_c", "choice_d",
                 "answer", "video_path"]:
        arrays.append(pa.array(cols[name], type=pa.string()))
        names.append(name)
    arrays.append(pa.array(cols["audio_path"], type=MEDIA_STRUCT))
    names.append("audio_path")
    arrays.append(pa.array(cols["image_path"], type=MEDIA_STRUCT))
    names.append("image_path")
    for name in ["task_type", "category", "duration",
                 "meta_audio_type", "meta_audio_content", "meta_image_content"]:
        arrays.append(pa.array(cols[name], type=pa.string()))
        names.append(name)

    table = pa.Table.from_arrays(arrays, names=names)
    hf_meta = {"info": {"features": {
        "audio_path": {"_type": "Audio"},
        "image_path": {"_type": "Image"},
    }}}
    table = table.replace_schema_metadata({"huggingface": json.dumps(hf_meta)})
    pq.write_table(table, out_path)
    print(f"[omnibench] written {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")


def build_daily_omni_parquet(out_path):
    src = os.path.join(LOCAL_ROOT, "data", "daily_omni.json")
    records = json.load(open(src, encoding="utf-8"))
    print(f"[daily_omni] records: {len(records)} (from {src})")

    field_names = [
        "id", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer",
        "video_path", "audio_path", "image_path", "task_type", "category", "duration",
        "meta_content_parent_category", "meta_content_fine_category",
    ]
    cols = {k: [] for k in field_names}
    for r in records:
        for k in field_names:
            if k in ("audio_path", "video_path", "image_path"):
                continue
            cols[k].append(r.get(k))
        cols["video_path"].append({"bytes": None, "path": hf_uri(r.get("video_path"))})
        cols["audio_path"].append({"bytes": None, "path": hf_uri(r.get("audio_path"))})
        cols["image_path"].append({"bytes": None, "path": hf_uri(r.get("image_path"))})

    arrays, names = [], []
    for name in ["id", "question", "choice_a", "choice_b", "choice_c", "choice_d", "answer"]:
        arrays.append(pa.array(cols[name], type=pa.string()))
        names.append(name)
    arrays.append(pa.array(cols["video_path"], type=MEDIA_STRUCT))
    names.append("video_path")
    arrays.append(pa.array(cols["audio_path"], type=MEDIA_STRUCT))
    names.append("audio_path")
    arrays.append(pa.array(cols["image_path"], type=MEDIA_STRUCT))
    names.append("image_path")
    for name in ["task_type", "category", "duration",
                 "meta_content_parent_category", "meta_content_fine_category"]:
        arrays.append(pa.array(cols[name], type=pa.string()))
        names.append(name)

    table = pa.Table.from_arrays(arrays, names=names)
    hf_meta = {"info": {"features": {
        "video_path": {"_type": "Video"},
        "audio_path": {"_type": "Audio"},
        "image_path": {"_type": "Image"},
    }}}
    table = table.replace_schema_metadata({"huggingface": json.dumps(hf_meta)})
    pq.write_table(table, out_path)
    print(f"[daily_omni] written {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")


NEW_README = """---
language:
- en
license: other
task_categories:
- question-answering
- visual-question-answering
tags:
- multimodal
- omni
- audio
- image
- video
- benchmark
size_categories:
- 1K<n<10K
configs:
- config_name: omnibench
  data_files:
  - split: train
    path: "data/omnibench-train-*.parquet"
  default: true
- config_name: daily_omni
  data_files:
  - split: train
    path: "data/daily_omni-train-*.parquet"
---

# Unify-OmniBench

> 统一格式的多模态评测数据集集合，由 Unify-OmniBench 框架转换生成。
> 目前包含 **OmniBench**（Image+Audio）与 **Daily-Omni**（Video+Audio）两个 bench，
> 在 Dataset Viewer 页面右上角可以通过下拉框在两者之间切换查看。

## 数据概览

| Subset (bench) | 题目数 | 模态 | 媒体 |
|---|---|---|---|
| `omnibench` | 1142 | Image + Audio | .png + .mp3 |
| `daily_omni` | 1197 | Video + Audio | .mp4 + .wav |

## 目录结构

```
├── data/
│   ├── omnibench-train-00000-of-00001.parquet    # omnibench subset（Dataset Viewer 用）
│   ├── daily_omni-train-00000-of-00001.parquet   # daily_omni subset（Dataset Viewer 用）
│   ├── omnibench.json                            # 原始 JSON（1142 条）
│   └── daily_omni.json                           # 原始 JSON（1197 条）
└── media/
    ├── image/omnibench/       # omnibench 图片
    ├── audio/omnibench/       # omnibench 音频
    ├── video/daily_omni/      # daily_omni 视频
    └── audio/daily_omni/      # daily_omni 音频
```

## 数据格式

### omnibench

```json
{
  "id": "omnibench:0",
  "question": "What are the men doing?",
  "choice_a": "...", "choice_b": "...", "choice_c": "...", "choice_d": "...",
  "answer": "C",
  "video_path": null,
  "audio_path": "media/audio/omnibench/2_009_four_people.mp3",
  "image_path": "media/image/omnibench/2_009_four_people.png",
  "task_type": "Action and Activity",
  "category": null,
  "duration": null,
  "meta_audio_type": "speech",
  "meta_audio_content": "...",
  "meta_image_content": "..."
}
```

### daily_omni

```json
{
  "id": "daily_omni:0:Ec_lQgZ9wlg",
  "question": "What visual elements were displayed immediately after ...?",
  "choice_a": "...", "choice_b": "...", "choice_c": "...", "choice_d": "...",
  "answer": "B",
  "video_path": "media/video/daily_omni/Ec_lQgZ9wlg.mp4",
  "audio_path": "media/audio/daily_omni/Ec_lQgZ9wlg.wav",
  "image_path": null,
  "task_type": "Event Sequence",
  "category": "Howto & Style",
  "duration": "30s",
  "meta_content_parent_category": "Lifestyle",
  "meta_content_fine_category": "Skincare Routines"
}
```

## 使用方式

```python
from datasets import load_dataset

omnibench = load_dataset("yukiyounai/Unify-OmniBench", "omnibench", split="train")
daily_omni = load_dataset("yukiyounai/Unify-OmniBench", "daily_omni", split="train")

print(omnibench[0]["question"], "→", omnibench[0]["answer"])
print(daily_omni[0]["question"], "→", daily_omni[0]["answer"])
```

## 引用

- **OmniBench**: [arXiv 2409.15272](https://arxiv.org/abs/2409.15272)
- **Daily-Omni**: [arXiv 2505.17862](https://arxiv.org/pdf/2505.17862) /
  [Lliar-liar/Daily-Omni](https://github.com/Lliar-liar/Daily-Omni)
- **转换工具**: Unify-OmniBench

## License

请参考原始 OmniBench / Daily-Omni 数据集的许可协议。
"""


def main():
    token = os.environ.get("HF_TOKEN")
    if not token and not DRY_RUN:
        sys.exit("请先 export HF_TOKEN=hf_xxx （需要对 yukiyounai/Unify-OmniBench 有写权限）")

    work = "/tmp/hf_push_work"
    os.makedirs(work, exist_ok=True)
    omnibench_parquet = os.path.join(work, "omnibench-train-00000-of-00001.parquet")
    daily_omni_parquet = os.path.join(work, "daily_omni-train-00000-of-00001.parquet")
    readme_path = os.path.join(work, "README.md")

    build_omnibench_parquet(omnibench_parquet)
    build_daily_omni_parquet(daily_omni_parquet)
    open(readme_path, "w", encoding="utf-8").write(NEW_README)
    print(f"[readme] written {readme_path}")

    video_dir = os.path.join(LOCAL_ROOT, "media", "video", "daily_omni")
    audio_dir = os.path.join(LOCAL_ROOT, "media", "audio", "daily_omni")
    n_video = len([f for f in os.listdir(video_dir) if f.endswith(".mp4")]) if os.path.isdir(video_dir) else 0
    n_audio = len([f for f in os.listdir(audio_dir) if f.endswith(".wav")]) if os.path.isdir(audio_dir) else 0
    print(f"[daily_omni media] video files: {n_video}, audio files: {n_audio}")

    if DRY_RUN:
        print("DRY_RUN=1，跳过实际上传，脚本自检结束。")
        return

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi(token=token)

    print("uploading media/video/daily_omni ...")
    api.upload_folder(
        repo_id=REPO_ID, repo_type="dataset", revision=REF,
        folder_path=video_dir, path_in_repo="media/video/daily_omni",
        commit_message="data: add Daily-Omni video files",
    )
    print("uploading media/audio/daily_omni ...")
    api.upload_folder(
        repo_id=REPO_ID, repo_type="dataset", revision=REF,
        folder_path=audio_dir, path_in_repo="media/audio/daily_omni",
        commit_message="data: add Daily-Omni audio files",
    )

    ops = [
        CommitOperationAdd(
            path_in_repo="data/omnibench-train-00000-of-00001.parquet",
            path_or_fileobj=omnibench_parquet,
        ),
        CommitOperationAdd(
            path_in_repo="data/daily_omni-train-00000-of-00001.parquet",
            path_or_fileobj=daily_omni_parquet,
        ),
        CommitOperationAdd(
            path_in_repo="data/daily_omni.json",
            path_or_fileobj=os.path.join(LOCAL_ROOT, "data", "daily_omni.json"),
        ),
        # 旧的单一 "default" config 文件，重构为按 bench 命名后删除
        CommitOperationDelete(path_in_repo="data/train-00000-of-00001.parquet"),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_path),
    ]
    res = api.create_commit(
        repo_id=REPO_ID, repo_type="dataset", revision=REF, operations=ops,
        commit_message=(
            "feat: add Daily-Omni subset, restructure into omnibench/daily_omni "
            "configs for per-bench Viewer selector"
        ),
    )
    print(res)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""构建规范 Parquet 文件，用于 HuggingFace Dataset Viewer 正确展示所有三个 Benchmark。

依赖：pip install -U pyarrow huggingface_hub

输出结构（/tmp/hf_parquet/）：
    omnibench-train-00000-of-00001.parquet
    daily_omni-train-00000-of-00001.parquet
    omnivideobench-train-00000-of-00001.parquet

每个 Parquet 的 audio_path / image_path 列都标记了 _type: Audio / Image 元数据，
Dataset Viewer 会渲染播放器和图片预览。

使用方式：
    # 1) 先确保 data/*.json 已由 convert_*.py 生成
    # 2) 干跑：只构建，不上传（验证 Parquet 正确性）
    DRY_RUN=1 python3 script/build_hf_parquet.py

    # 3) 正式推送
    export HF_TOKEN=hf_xxx
    python3 script/build_hf_parquet.py

可选环境变量：
    LOCAL_ROOT   本地数据根目录
    REPO_ID      HF 仓库 id（默认 yukiyounai/Unify-OmniBench）
    DRY_RUN=1    只构建不上传
"""

import json
import os
import sys
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

# ── 可配置项 ──────────────────────────────────────────────
REPO_ID = os.environ.get("REPO_ID", "yukiyounai/Unify-OmniBench")
REF = "main"
LOCAL_ROOT = os.environ.get(
    "LOCAL_ROOT",
    "/apdcephfs_hldy/share_304318596/weiyangguo/Datasets/Unify-OmniBench",
)
DRY_RUN = os.environ.get("DRY_RUN") == "1"
WORK_DIR = "/tmp/hf_parquet"

# 统一的 Audio/Image struct 类型（格式：{bytes: null, path: "hf://..."}）
MEDIA_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


# ── 工具函数 ──────────────────────────────────────────────
def hf_uri(rel_path: Optional[str]) -> Optional[str]:
    if not rel_path:
        return None
    return f"hf://datasets/{REPO_ID}@{REF}/{rel_path}"


def flatten_choices(choices_list: list, letters: List[str] = None) -> dict:
    """将 choices: ["A. xxx", "B. yyy", ...] 拍平为 {"choice_a":"xxx", ...}"""
    if letters is None:
        letters = list("ABCD")
    out = {f"choice_{l.lower()}": None for l in letters}
    for c in choices_list:
        c = str(c).strip()
        for i, l in enumerate(letters):
            prefix = f"{l}." if c.startswith(f"{l}.") else None
            prefix2 = f"{l})" if c.startswith(f"{l})") else None
            if prefix or prefix2:
                out[f"choice_{l.lower()}"] = c[2:].strip()
                break
        else:
            # 没有字母前缀，按顺序填入
            for i, l in enumerate(letters):
                if out[f"choice_{l.lower()}"] is None:
                    out[f"choice_{l.lower()}"] = c
                    break
    return out


def flatten_meta(meta: Optional[dict]) -> dict:
    """将 meta dict 拍平为 meta_xxx 列前缀。取前 10 个 key 避免列膨胀。"""
    if not meta:
        return {}
    out = {}
    for k, v in list(meta.items())[:10]:
        key = f"meta_{k}"
        if isinstance(v, (str, int, float, bool)) and not isinstance(v, bool):
            out[key] = str(v)
        elif v is None:
            out[key] = None
        else:
            out[key] = json.dumps(v, ensure_ascii=False)
    return out


def load_json(name: str) -> list:
    path = os.path.join(LOCAL_ROOT, "data", f"{name}.json")
    if not os.path.exists(path):
        print(f"[skip] {path} not found — please run convert_{name}.py first")
        return []
    records = json.load(open(path, encoding="utf-8"))
    print(f"[{name}] {len(records)} records loaded from {path}")
    return records


def build_parquet(
    name: str,
    records: list,
    media_cols: List[str],          # e.g. ["audio_path", "image_path"]
    meta_keys: Optional[List[str]] = None,  # extra meta keys to flatten
) -> str:
    """构建单个 benchmark 的 parquet，返回输出路径。"""
    out = os.path.join(WORK_DIR, f"{name}-train-00000-of-00001.parquet")
    letters = list("ABCD")

    # 第一遍扫描：收集所有出现的 meta key（拍平后的列名）
    all_meta_keys = set()
    for r in records:
        meta = r.get("meta") or {}
        for k in meta:
            all_meta_keys.add(f"meta_{k}")
    meta_key_list = sorted(all_meta_keys)[:10]  # 最多 10 个 meta 列

    # 列定义
    core_cols = ["id", "question", "answer", "task_type", "category", "duration"]
    choice_cols = [f"choice_{l.lower()}" for l in letters]

    # 初始化
    cols: dict = {k: [] for k in core_cols + choice_cols + meta_key_list}
    for mc in media_cols:
        cols[mc] = []

    for r in records:
        for ck in core_cols:
            val = r.get(ck)
            cols[ck].append(str(val) if val is not None else None)

        choices_flat = flatten_choices(r.get("choices") or [], letters)
        for cl in choice_cols:
            cols[cl].append(choices_flat.get(cl))

        meta_flat = flatten_meta(r.get("meta"))
        for mk in meta_key_list:
            cols[mk].append(meta_flat.get(mk))

        for mc in media_cols:
            rel = r.get(mc)
            cols[mc].append({"bytes": None, "path": hf_uri(rel)} if rel else None)

    # 构造 Arrow Table
    arrays, names = [], []
    for ck in core_cols:
        arrays.append(pa.array(cols[ck], type=pa.string()))
        names.append(ck)
    for cl in choice_cols:
        arrays.append(pa.array(cols[cl], type=pa.string()))
        names.append(cl)
    for mc in media_cols:
        arrays.append(pa.array(cols[mc], type=MEDIA_STRUCT))
        names.append(mc)
    for mk in meta_key_list:
        arrays.append(pa.array(cols[mk], type=pa.string()))
        names.append(mk)

    table = pa.Table.from_arrays(arrays, names=names)

    # 标记 Audio / Image 类型
    feature_types = {}
    for mc in media_cols:
        if "audio" in mc:
            feature_types[mc] = {"_type": "Audio"}
        elif "image" in mc:
            feature_types[mc] = {"_type": "Image"}
    if feature_types:
        hf_meta = {"info": {"features": feature_types}}
        table = table.replace_schema_metadata({"huggingface": json.dumps(hf_meta)})

    pq.write_table(table, out)
    size_mb = os.path.getsize(out) / 1e6
    print(f"  -> {out} ({size_mb:.2f} MB, {len(records)} rows, "
          f"{len(table.column_names)} cols)")
    return out


# ── README 模板 ───────────────────────────────────────────
README_TEXT = """---
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
- config_name: daily_omni
  data_files:
  - split: train
    path: "data/daily_omni-train-*.parquet"
  default: true
- config_name: omnibench
  data_files:
  - split: train
    path: "data/omnibench-train-*.parquet"
- config_name: omnivideobench
  data_files:
  - split: train
    path: "data/omnivideobench-train-*.parquet"
---

# Unify-OmniBench

统一格式的多模态评测数据集，由 [Unify-OmniBench](https://github.com/xxx) 框架转换生成。
包含三个 benchmark，在 Dataset Viewer 右上角下拉框切换。

## 数据概览

| Config (bench) | 题目数 | 模态 | 媒体 |
|---|---|---|---|
| `daily_omni` | ~1197 | Video + Audio | .mp4 + .wav |
| `omnibench` | ~1142 | Image + Audio | .png/.jpg + .mp3 |
| `omnivideobench` | ~1000 | Video (embedded audio) | .mp4 |

## 使用方式

```python
from datasets import load_dataset

ds = load_dataset("REPO_ID_PLACEHOLDER", "daily_omni", split="train")
print(ds[0]["question"], "->", ds[0]["answer"])
```

## 引用

- **OmniBench**: [arXiv 2409.15272](https://arxiv.org/abs/2409.15272)
- **Daily-Omni**: [arXiv 2505.17862](https://arxiv.org/abs/2505.17862)
- **OmniVideoBench**: [NJU-LINK/OmniVideoBench](https://github.com/NJU-LINK/OmniVideoBench)
- **转换工具**: Unify-OmniBench
"""

# ── 主流程 ────────────────────────────────────────────────
def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    # 三个 benchmark 的配置
    benchmarks = {
        "daily_omni": {
            "media_cols": ["video_path", "audio_path"],
        },
        "omnibench": {
            "media_cols": ["image_path", "audio_path"],
        },
        "omnivideobench": {
            "media_cols": ["video_path"],
        },
    }

    parquet_files = {}
    for name, cfg in benchmarks.items():
        records = load_json(name)
        if not records:
            print(f"[warn] {name}: no data, skipping")
            continue
        parquet_files[name] = build_parquet(
            name=name,
            records=records,
            media_cols=cfg["media_cols"],
        )

    if not parquet_files:
        sys.exit("No data files found. Run conversion scripts first.")

    # 写 README
    readme_text = README_TEXT.replace("REPO_ID_PLACEHOLDER", REPO_ID)
    readme_path = os.path.join(WORK_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_text)
    print(f"[readme] {readme_path}")

    if DRY_RUN:
        print("\n✅ DRY_RUN=1 — Parquet 文件已生成在 /tmp/hf_parquet/，"
              "未实际上传。检查无误后去掉 DRY_RUN 重新运行。")
        return

    # ── 上传 ──
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("请先 export HF_TOKEN=hf_xxx（需要对 {} 有写权限）".format(REPO_ID))

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)

    # 验证 token
    try:
        who = api.whoami()
        print(f"[auth] logged in as: {who.get('name', '?')}")
    except Exception as e:
        sys.exit(f"认证失败: {e}\n"
                 "  检查: 1) HF_TOKEN 是否正确 2) token 是否过期 3) 是否有 {REPO_ID} 的写权限")

    # 上传媒体文件夹（默认跳过；设置 WITH_MEDIA=1 才会传）
    if os.environ.get("WITH_MEDIA") == "1":
        media_base = os.path.join(LOCAL_ROOT, "media")
        if os.path.isdir(media_base):
            for sub in os.listdir(media_base):
                sub_path = os.path.join(media_base, sub)
                if not os.path.isdir(sub_path):
                    continue
                # 统计递归文件数
                file_count = sum(
                    1 for _ in os.listdir(sub_path)
                    if os.path.isfile(os.path.join(sub_path, _)) and not _.startswith('.')
                )
                if file_count == 0:
                    continue
                target = f"media/{sub}"
                print(f"[upload] {target}/ ({file_count} files) ...")
                api.upload_large_folder(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    revision=REF,
                    folder_path=sub_path,
                    path_in_repo=target,
                )
    else:
        print("[skip] media upload (set WITH_MEDIA=1 to upload)")

    # 上传 Parquet + README（一个 commit）
    ops = []
    for name, pf in parquet_files.items():
        ops.append(CommitOperationAdd(
            path_in_repo=f"data/{name}-train-00000-of-00001.parquet",
            path_or_fileobj=pf,
        ))
    # 同时上传原始 JSON 留底
    for name in parquet_files:
        json_path = os.path.join(LOCAL_ROOT, "data", f"{name}.json")
        if os.path.exists(json_path):
            ops.append(CommitOperationAdd(
                path_in_repo=f"data/{name}.json",
                path_or_fileobj=json_path,
            ))
    ops.append(CommitOperationAdd(
        path_in_repo="README.md",
        path_or_fileobj=readme_path,
    ))

    print(f"[commit] pushing {len(ops)} files to {REPO_ID}@{REF} ...")
    res = api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REF,
        operations=ops,
        commit_message=(
            "feat: add canonical parquet for daily_omni/omnibench/omnivideobench "
            "+ omnivideobench subset"
        ),
    )
    print(f"[commit] done: {res}")
    print("\n✅ 推送完成。等几分钟后刷新 Dataset Viewer 页面查看效果。")


if __name__ == "__main__":
    main()

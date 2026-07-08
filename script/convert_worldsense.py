#!/usr/bin/env python3
"""Convert WorldSense (VLMEvalKit format) → unified JSON format.

Source layout (as produced by VLMEvalKit's
``vlmeval/dataset/worldsense.py::WorldSense.prepare_dataset()``, i.e. the
directory pointed to by ``$WORLDSENSE_DATA_DIR`` after a run of
``VLMEvalKit/eval.sh``)::

    <data_dir>/
      WorldSense.tsv          # one row per QA pair (tab-separated)
      videos/{video}.mp4
      audios/{video}.wav      # moviepy-extracted from videos/{video}.mp4
      subtitles/{video}.srt   # not migrated (Unify has no subtitle modality)

TSV columns (see ``worldsense.py::generate_tsv``)::

    index, video, video_path, duration, domain, candidates, sub_category,
    audio_class, task_domain, task_type, subtitle_path, audio_path,
    video_caption, question, answer

Notes on field encoding (both are Python-repr'd list *strings*, need ``eval``):
  * ``candidates``  -> ``"['A. foo', 'B. bar', ...]"``
  * ``audio_class`` -> ``"['Speech']"`` (can have multiple entries)

Usage:
    python script/convert_worldsense.py \
        --data-dir /path/to/WorldSense \
        --out-dir /path/to/Unify-OmniBench/data/root

Output:
    {out_dir}/data/worldsense.json
    {out_dir}/media/video/worldsense/{video}.mp4
    (audios/{video}.wav 不复制——只用交织模式，音频直接从 .mp4 容器提取；
    见 dataset_config.yaml::worldsense.use_audio_in_video)
"""
import argparse
import ast
import json
import os
import shutil


def _safe_eval_list(s, default=None):
    """Parse a Python-repr'd list string (e.g. "['A. foo', 'B. bar']")."""
    if s is None:
        return default if default is not None else []
    if isinstance(s, list):
        return s
    try:
        v = ast.literal_eval(str(s))
        return v if isinstance(v, list) else [v]
    except Exception:
        return default if default is not None else [str(s)]


def main():
    p = argparse.ArgumentParser(description="Convert WorldSense (VLMEvalKit) to unified format")
    p.add_argument("--data-dir", required=True,
                    help="WorldSense root dir containing WorldSense.tsv + videos/ + audios/")
    p.add_argument("--out-dir", required=True, help="Output root directory")
    p.add_argument("--tsv-name", default="WorldSense.tsv")
    args = p.parse_args()

    import pandas as pd  # local import — keep script runnable without pandas at import time

    tsv_path = os.path.join(args.data_dir, args.tsv_name)
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"Loaded {len(df)} rows from {tsv_path}")

    data_dir = os.path.join(args.out_dir, "data")
    video_out = os.path.join(args.out_dir, "media", "video", "worldsense")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(video_out, exist_ok=True)
    # 只用交织模式（use_audio_in_video=True），不复制独立 audios/{v}.wav——
    # 音频推理时直接从 .mp4 容器提取，避免同一段音频被喂两遍。

    media_status = {}  # video_name -> video_rel_or_None
    records = []

    total = len(df)
    for i, row in df.iterrows():
        if (i + 1) % 200 == 0 or i + 1 == total:
            print(f"[{i + 1}/{total}]", flush=True)

        video_name = str(row["video"])
        if video_name not in media_status:
            # video_path column is "./videos/{v}.mp4"
            src_video = os.path.join(args.data_dir, str(row["video_path"]).lstrip("./"))
            v_rel = None
            if os.path.exists(src_video):
                dst_video = os.path.join(video_out, f"{video_name}.mp4")
                if not os.path.exists(dst_video):
                    shutil.copy2(src_video, dst_video)
                v_rel = f"media/video/worldsense/{video_name}.mp4"
            media_status[video_name] = v_rel
        video_rel = media_status[video_name]

        choices = _safe_eval_list(row.get("candidates"))
        audio_class = _safe_eval_list(row.get("audio_class"))

        records.append({
            "id": f"worldsense:{row.get('index', i)}",
            "question": row.get("question", ""),
            "choices": choices,
            "answer": row.get("answer"),
            "video_path": video_rel,
            # 只用交织模式：不挂独立 audio，音频从视频容器里提取（见
            # dataset_config.yaml::worldsense.use_audio_in_video=true）。
            "audio_path": None,
            "image_path": None,
            "task_type": row.get("task_type"),
            "category": row.get("domain"),
            "duration": row.get("duration"),  # already a bucket string, e.g. "<1min"
            "meta": {
                "video": video_name,
                "domain": row.get("domain"),
                "sub_category": row.get("sub_category"),
                "task_domain": row.get("task_domain"),
                "audio_class": audio_class,          # list, e.g. ["Speech"]
                "duration_category": row.get("duration"),
                "video_caption": row.get("video_caption"),
            },
        })

    out_json = os.path.join(data_dir, "worldsense.json")
    json.dump(records, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n_video = sum(1 for v in media_status.values() if v)
    n_total_videos = len(media_status)
    print(f"Done: {out_json} ({len(records)} QA records, {n_total_videos} unique videos)")
    print(f"  video: {n_video}/{n_total_videos} copied; audio not copied (interleaved-only)")


if __name__ == "__main__":
    main()

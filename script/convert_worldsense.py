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
    {out_dir}/media/audio/worldsense/{video}.wav
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
    audio_out = os.path.join(args.out_dir, "media", "audio", "worldsense")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(video_out, exist_ok=True)
    os.makedirs(audio_out, exist_ok=True)
    # [2026-07-07 REVERTED] Previously this script dropped the independent
    # audios/{v}.wav and relied solely on use_audio_in_video=True to extract
    # audio from the .mp4 container (interleaved mode), reasoning it was a
    # redundant duplicate of the same content. CONFIRMED BROKEN IN PRACTICE:
    # running worldsense with use_audio_in_video=True on pinned vllm==0.11.0
    # crashes the V1 engine's GPU worker on every single sample with
    #   RuntimeError: Worker failed with error 'index 1 is out of bounds
    #   for dimension 0 with size 1'
    # — this is exactly the interleaved-modality limitation already flagged
    # (but never tested) in vllm_runner.py's module docstring: "V1 engine
    # does not support interleaved modalities yet". Reverted to copying the
    # independent .wav + use_audio_in_video=False (see dataset_config.yaml),
    # matching the officially-supported "mixed_modalities" pattern already
    # working for Daily-Omni/OmniBench.

    media_status = {}  # video_name -> (video_rel_or_None, audio_rel_or_None)
    records = []

    total = len(df)
    for i, row in df.iterrows():
        if (i + 1) % 200 == 0 or i + 1 == total:
            print(f"[{i + 1}/{total}]", flush=True)

        video_name = str(row["video"])
        if video_name not in media_status:
            # video_path / audio_path columns are "./videos/{v}.mp4" / "./audios/{v}.wav"
            src_video = os.path.join(args.data_dir, str(row["video_path"]).lstrip("./"))
            src_audio = os.path.join(args.data_dir, str(row["audio_path"]).lstrip("./"))
            v_rel = None
            a_rel = None
            if os.path.exists(src_video):
                dst_video = os.path.join(video_out, f"{video_name}.mp4")
                if not os.path.exists(dst_video):
                    shutil.copy2(src_video, dst_video)
                v_rel = f"media/video/worldsense/{video_name}.mp4"
            if os.path.exists(src_audio):
                dst_audio = os.path.join(audio_out, f"{video_name}.wav")
                if not os.path.exists(dst_audio):
                    shutil.copy2(src_audio, dst_audio)
                a_rel = f"media/audio/worldsense/{video_name}.wav"
            media_status[video_name] = (v_rel, a_rel)
        video_rel, audio_rel = media_status[video_name]

        choices = _safe_eval_list(row.get("candidates"))
        audio_class = _safe_eval_list(row.get("audio_class"))

        records.append({
            "id": f"worldsense:{row.get('index', i)}",
            "question": row.get("question", ""),
            "choices": choices,
            "answer": row.get("answer"),
            "video_path": video_rel,
            # [2026-07-07 REVERTED to independent audio, see the long note
            # above main()'s media-copy loop] use_audio_in_video=True
            # (interleaved) is confirmed broken on pinned vllm==0.11.0's V1
            # engine — crashes every sample with "index 1 is out of bounds
            # for dimension 0 with size 1". Back to attaching the
            # independent .wav + use_audio_in_video=False in
            # dataset_config.yaml.
            "audio_path": audio_rel,
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

    n_video = sum(1 for v, a in media_status.values() if v)
    n_audio = sum(1 for v, a in media_status.values() if a)
    n_total_videos = len(media_status)
    print(f"Done: {out_json} ({len(records)} QA records, {n_total_videos} unique videos)")
    print(f"  video: {n_video}/{n_total_videos} copied, audio: {n_audio}/{n_total_videos} copied")


if __name__ == "__main__":
    main()

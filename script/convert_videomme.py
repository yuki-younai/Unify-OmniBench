#!/usr/bin/env python3
"""Convert Video-MME (VLMEvalKit format) → unified JSON format.

Source layout (as produced by VLMEvalKit's
``vlmeval/dataset/videomme.py::VideoMME.generate_tsv()``)::

    <data_dir>/
      Video-MME.tsv           # one row per QA pair (tab-separated)
      video/{videoID}.mp4
      subtitle/{videoID}.srt  # not migrated (Unify has no subtitle modality)

TSV columns (see ``videomme.py::generate_tsv``)::

    index, video, video_path, duration, domain, candidates,
    sub_category, task_type, subtitle_path, question, answer

Note on ``candidates`` field: stored as Python-repr'd list string
(e.g. "['A. foo', 'B. bar', 'C. baz', 'D. qux']"), needs ``eval``.

Usage:
    python script/convert_videomme.py \\
        --data-dir /path/to/Video-MME \\
        --out-dir /path/to/Unify-OmniBench/data/root

Output:
    {out_dir}/data/videomme.json
    {out_dir}/media/video/videomme/{videoID}.mp4
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
    p = argparse.ArgumentParser(description="Convert Video-MME (VLMEvalKit) to unified format")
    p.add_argument("--data-dir", required=True,
                   help="Video-MME root dir containing Video-MME.tsv + video/*.mp4")
    p.add_argument("--out-dir", required=True, help="Output root directory")
    p.add_argument("--tsv-name", default="Video-MME.tsv")
    args = p.parse_args()

    import pandas as pd

    tsv_path = os.path.join(args.data_dir, args.tsv_name)
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"Loaded {len(df)} rows from {tsv_path}")

    data_dir = os.path.join(args.out_dir, "data")
    video_out = os.path.join(args.out_dir, "media", "video", "videomme")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(video_out, exist_ok=True)

    # Video-MME is pure video (no audio tracks)
    # -> audio_path always None, use_audio_in_video=False

    media_status = {}  # video_name -> video_rel_or_None
    records = []

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 200 == 0 or i + 1 == total:
            print(f"[{i + 1}/{total}]", flush=True)

        video_name = str(row["video"])

        # Copy video file (dedup by video name)
        if video_name not in media_status:
            # video_path column is "./video/{v}.mp4"
            src_video = os.path.join(args.data_dir, str(row["video_path"]).lstrip("./"))
            v_rel = None
            if os.path.exists(src_video):
                dst_video = os.path.join(video_out, f"{video_name}.mp4")
                if not os.path.exists(dst_video):
                    shutil.copy2(src_video, dst_video)
                v_rel = f"media/video/videomme/{video_name}.mp4"
            media_status[video_name] = v_rel
        video_rel = media_status[video_name]

        choices = _safe_eval_list(row.get("candidates"))

        records.append({
            "id": f"videomme:{row.get('index', i)}",
            "question": row.get("question", ""),
            "choices": choices,
            "answer": row.get("answer"),   # already A/B/C/D
            "video_path": video_rel,
            "audio_path": None,            # Video-MME has no audio
            "image_path": None,
            "task_type": row.get("task_type"),
            "category": row.get("domain"),
            "duration": row.get("duration"),  # "short" / "medium" / "long"
            "meta": {
                "video": video_name,
                "domain": row.get("domain"),
                "sub_category": row.get("sub_category"),
                "duration_category": row.get("duration"),
            },
        })

    out_json = os.path.join(data_dir, "videomme.json")
    json.dump(records, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n_video = sum(1 for v in media_status.values() if v)
    n_total_videos = len(media_status)
    print(f"Done: {out_json} ({len(records)} QA records, {n_total_videos} unique videos)")
    print(f"  video: {n_video}/{n_total_videos} copied; audio not applicable (pure-video dataset)")


if __name__ == "__main__":
    main()

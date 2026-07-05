#!/usr/bin/env python3
"""Convert OmniVideoBench → unified JSON format.

Supports both raw data layouts seen in ``OmniVideoBench/dataloader.py``:
  * Nested schema (``data.json``): one item per video, with a
    ``questions`` list (each question has ``options`` / ``correct_option``
    / ``question_type`` / ``audio_type`` ...).
  * Flat schema (HF parquet export, e.g. ``data.parquet``): one row per
    QA pair, with columns ``video`` / ``duration`` / ``question`` /
    ``options`` / ``correct_option``. The ``video`` value may already
    include a relative path + extension (e.g. ``videos/video_1.mp4``).

Usage:
    python script/convert_omnivideobench.py \
        --data-file /path/to/data.parquet \
        --video-dir /path/to/OmniVideoBench \
        --out-dir /path/to/output

Output:
    {out_dir}/data/omnivideobench.json
    {out_dir}/media/video/omnivideobench/{video_name}.mp4
"""
import argparse, json, os, shutil


def _load_data(path: str):
    if path.endswith(".parquet"):
        import pandas as pd  # type: ignore
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_list(options):
    """pandas may hand back numpy arrays for list-typed columns."""
    if hasattr(options, "tolist"):
        return options.tolist()
    return options if isinstance(options, list) else ([] if options is None else [options])


def _video_rel_and_name(video_field: str, default_ext: str = ".mp4") -> tuple:
    """Resolve a raw ``video`` field to (rel_path_from_video_dir, bare_name).

    Mirrors ``OmniVideoBench/dataloader.py``:
      * If the field already ends with a video extension (e.g.
        ``"videos/video_1.mp4"``), it's used AS-IS relative to
        ``video_dir`` (preserving any subdirectory like ``videos/``).
      * Otherwise (bare id, e.g. ``"video_1"``) append ``default_ext``
        and assume it sits directly under ``video_dir`` (no subdir).
    """
    v = str(video_field)
    if v.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")):
        rel = v
    else:
        rel = v + default_ext
    name = os.path.splitext(os.path.basename(rel))[0]
    return rel, name


def _letter_from_options(answer_text, options):
    """Fallback: derive A/B/C/D from full-text answer by matching options."""
    if not answer_text:
        return None
    a = str(answer_text).strip()
    if len(a) == 1 and a.upper() in "ABCDEFGHIJ":
        return a.upper()
    letters = list("ABCDEFGHIJ")
    for i, opt in enumerate(options or []):
        o = str(opt).strip()
        # strip leading "A." / "A. " / "A)" prefix for comparison
        if len(o) > 1 and o[0].upper() in letters and o[1] in ".) ":
            o_body = o[2:].strip()
        else:
            o_body = o
        if o_body == a:
            return letters[i]
    return a.upper()[:1] if a else None


def _extract_rows(data):
    """Yield normalized dicts: {video_rel, video_name, video_type, duration,
    question, options, answer_letter, question_type, audio_type}."""
    for item in data:
        if isinstance(item, dict) and "questions" in item and "question" not in item:
            # nested schema
            vrel, vname = _video_rel_and_name(item.get("video", "unknown_video"))
            for qa in item.get("questions", []):
                options = _to_list(qa.get("options"))
                yield {
                    "video_rel": vrel,
                    "video_name": vname,
                    "video_type": item.get("video_type"),
                    "duration": item.get("duration"),
                    "question": qa.get("question", ""),
                    "options": options,
                    "answer_letter": _letter_from_options(
                        qa.get("correct_option") or qa.get("answer"), options
                    ),
                    "question_type": qa.get("question_type"),
                    "audio_type": qa.get("audio_type"),
                }
        else:
            # flat schema (one row == one QA pair)
            options = _to_list(item.get("options"))
            vrel, vname = _video_rel_and_name(item.get("video", "unknown_video"))
            yield {
                "video_rel": vrel,
                "video_name": vname,
                "video_type": item.get("video_type"),
                "duration": item.get("duration"),
                "question": item.get("question", ""),
                "options": options,
                "answer_letter": _letter_from_options(
                    item.get("correct_option") or item.get("answer"), options
                ),
                "question_type": item.get("question_type"),
                "audio_type": item.get("audio_type"),
            }


def main():
    p = argparse.ArgumentParser(description="Convert OmniVideoBench to unified format")
    p.add_argument("--data-file", required=True, help="Path to data.json or data.parquet")
    p.add_argument("--video-dir", required=True, help="Directory containing the .mp4 files")
    p.add_argument("--out-dir", required=True, help="Output root directory")
    p.add_argument("--video-ext", default=".mp4")
    args = p.parse_args()

    data_dir = os.path.join(args.out_dir, "data")
    video_out = os.path.join(args.out_dir, "media", "video", "omnivideobench")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(video_out, exist_ok=True)

    raw = _load_data(args.data_file)
    rows = list(_extract_rows(raw))
    total = len(rows)
    print(f"Parsed {total} QA pairs from {args.data_file}")

    records = []
    video_status = {}  # vname -> relative unified path, or None if missing

    for i, r in enumerate(rows):
        missing_v = sum(1 for v in video_status.values() if v is None)
        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"[{i + 1}/{total}] unique_videos={len(video_status)} missing_video={missing_v}",
                  flush=True)

        vname = r["video_name"]
        if vname not in video_status:
            src = os.path.join(args.video_dir, r["video_rel"])
            dst = os.path.join(video_out, f"{vname}{args.video_ext}")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                video_status[vname] = f"media/video/omnivideobench/{vname}{args.video_ext}"
            else:
                video_status[vname] = None
        vpath = video_status[vname]

        options = [str(o).strip() for o in (r["options"] or [])]

        records.append({
            "id": f"omnivideobench:{i}:{vname}",
            "question": r["question"],
            "choices": options,
            "answer": r["answer_letter"],
            "video_path": vpath,
            "audio_path": None,
            "image_path": None,
            "task_type": r.get("question_type"),
            "category": r.get("video_type"),
            "duration": r.get("duration"),
            "meta": {
                "video": vname,
                "audio_type": r.get("audio_type"),
            },
        })

    out_json = os.path.join(data_dir, "omnivideobench.json")
    json.dump(records, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n_ok = sum(1 for v in video_status.values() if v is not None)
    n_total_videos = len(video_status)
    print(f"Done: {out_json} ({len(records)} records, {n_total_videos} unique videos)")
    print(f"  video: {n_ok}/{n_total_videos} unique videos copied "
          f"({n_total_videos - n_ok} missing)")


if __name__ == "__main__":
    main()

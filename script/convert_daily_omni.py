#!/usr/bin/env python3
"""Convert Daily-Omni → unified JSON format.

Usage:
    python script/convert_daily_omni.py \
        --qa-file /path/to/qa.json \
        --video-dir /path/to/Videos \
        --out-dir /path/to/output

Output:
    {out_dir}/data/daily_omni.json
    {out_dir}/media/video/daily_omni/{video_id}.mp4
    {out_dir}/media/audio/daily_omni/{video_id}.wav
"""
import argparse, json, os, shutil


def main():
    p = argparse.ArgumentParser(description="Convert Daily-Omni to unified format")
    p.add_argument("--qa-file", required=True, help="Path to qa.json")
    p.add_argument("--video-dir", required=True, help="Path to Videos/ directory")
    p.add_argument("--out-dir", required=True, help="Output root directory")
    args = p.parse_args()

    data_dir = os.path.join(args.out_dir, "data")
    video_out = os.path.join(args.out_dir, "media", "video", "daily_omni")
    audio_out = os.path.join(args.out_dir, "media", "audio", "daily_omni")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(video_out, exist_ok=True)
    os.makedirs(audio_out, exist_ok=True)

    data = json.load(open(args.qa_file, "r", encoding="utf-8"))
    print(f"Records: {len(data)}")

    records = []
    seen = set()
    missing_v = missing_a = 0

    for idx, item in enumerate(data):
        vid = str(item["video_id"])

        if vid not in seen:
            seen.add(vid)
            src_v = os.path.join(args.video_dir, vid, f"{vid}_video.mp4")
            src_a = os.path.join(args.video_dir, vid, f"{vid}_audio.wav")
            dst_v = os.path.join(video_out, f"{vid}.mp4")
            dst_a = os.path.join(audio_out, f"{vid}.wav")

            if os.path.exists(src_v):
                shutil.copy2(src_v, dst_v)
            else:
                missing_v += 1
            if os.path.exists(src_a):
                shutil.copy2(src_a, dst_a)
            else:
                missing_a += 1

        choices = item.get("Choice") or []
        if isinstance(choices, str):
            choices = [c.strip() for c in choices.split("\n") if c.strip()]

        records.append({
            "id": f"daily_omni:{idx}:{vid}",
            "question": item.get("Question", ""),
            "choices": choices,
            "answer": item.get("Answer", ""),
            "video_path": f"media/video/daily_omni/{vid}.mp4",
            "audio_path": f"media/audio/daily_omni/{vid}.wav",
            "image_path": None,
            "task_type": item.get("Type"),
            "category": item.get("video_category"),
            "duration": item.get("video_duration"),
            "meta": {
                "content_parent_category": item.get("content_parent_category"),
                "content_fine_category": item.get("content_fine_category"),
            },
        })

    out_json = os.path.join(data_dir, "daily_omni.json")
    json.dump(records, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"Done: {out_json} ({len(records)} records)")
    print(f"  video: {len(seen) - missing_v}/{len(seen)} copied")
    print(f"  audio: {len(seen) - missing_a}/{len(seen)} copied")


if __name__ == "__main__":
    main()

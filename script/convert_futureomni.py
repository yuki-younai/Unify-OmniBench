"""FutureOmni → Unify-OmniBench 统一格式转换。

用法:
    python script/convert_futureomni.py \
        --data-dir /path/to/FutureOmni \
        --out-dir /path/to/Unify-OmniBench

输入:
    <data-dir>/
      annotations.json (或 .jsonl)    # 原始题目
      videos/                          # 原始视频

输出:
    <out-dir>/
      data/future_omni.json
      media/video/future_omni/         # 截断到 split_point 的视频 (.mp4)

split_point 处理:
    原始视频在 split_point 帧处用 ffmpeg -c copy 流拷贝截断（不重编码），
    截断后的视频作为评测输入。模型只能看到 split_point 之前的音画内容，
    需要预测之后发生的事件。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def load_records(path: str):
    """Load records from JSON or JSONL."""
    with open(path, encoding="utf-8") as fp:
        text = fp.read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def truncate_video(src: str, dst: str, seconds: float) -> bool:
    """ffmpeg -c copy 截断视频到 seconds 秒。返回是否成功。"""
    if seconds <= 0:
        return False
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src, "-t", f"{seconds:.3f}",
        "-c", "copy", dst,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg error: {e}")
        return False


def extract_videos(data_dir: str) -> str:
    """合并分卷包并解压，返回解压后的 videos/ 路径。

    FutureOmni 发布格式:
        test_splitted_videos_part_aa, _ab, _ac → cat → .zip → unzip → videos/
    """
    parts = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("test_splitted_videos_part_")
    )
    if not parts:
        return os.path.join(data_dir, "videos")

    archive_path = os.path.join(data_dir, "test_splitted_videos.zip")
    video_dir = os.path.join(data_dir, "videos")

    if os.path.isdir(video_dir) and any(f.endswith(".mp4") for f in os.listdir(video_dir)):
        return video_dir
    # 视频可能已被解压到 data_dir 根目录（不在 videos/ 子目录）
    if any(f.endswith(".mp4") for f in os.listdir(data_dir)):
        return data_dir

    print("合并分卷包 ...")
    with open(archive_path, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                out.write(f.read())
    print(f"  → {archive_path} ({os.path.getsize(archive_path)/1024/1024:.0f} MB)")

    # 尝试 unzip，失败则试 tar
    print("解压 ...")
    result = subprocess.run(["unzip", "-o", archive_path, "-d", data_dir], capture_output=True)
    if result.returncode != 0:
        out = (result.stderr or result.stdout).decode(errors="replace")[:200]
        if "not a valid zip" in out.lower() or "End-of-central-directory" in out:
            print(f"  unzip 失败, 尝试 tar ...")
            subprocess.run(["tar", "-xf", archive_path, "-C", data_dir], check=True)
        else:
            print(f"  unzip 失败: {out}")
            result.check_returncode()
    os.remove(archive_path)
    return video_dir


def main():
    p = argparse.ArgumentParser(description="FutureOmni → Unify-OmniBench 转换")
    p.add_argument("--data-dir", required=True,
                   help="FutureOmni 原始数据目录（含 futureomni_test.json + 分卷包）")
    p.add_argument("--out-dir", required=True,
                   help="Unify-OmniBench 数据根目录")
    args = p.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir

    # 1) 合并分卷包 → 解压 videos/
    video_src_dir = extract_videos(data_dir)

    # 2) 加载标注
    anno_path = os.path.join(data_dir, "futureomni_test.json")
    if not os.path.exists(anno_path):
        sys.exit(f"标注文件不存在: {anno_path}")
    records = load_records(anno_path)
    print(f"加载: {len(records)} records from {anno_path}")

    video_dst_dir = os.path.join(out_dir, "media", "video", "future_omni")
    os.makedirs(video_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)

    samples = []
    skipped = 0

    for r in records:
        vid_name = r.get("video", "")
        # 视频文件按 qid 命名: 0.mp4, 1.mp4, ...
        src = os.path.join(video_src_dir, f"{r['qid']}.mp4")
        if not os.path.exists(src):
            print(f"  SKIP video not found: {vid_name}")
            skipped += 1
            continue

        # FutureOmni 用 seconds 而非 split_point（单位：秒，直接可用）
        seconds = float(r.get("seconds", 0))

        dst_name = f"{r['qid']}_premise.mp4"
        dst = os.path.join(video_dst_dir, dst_name)

        if not os.path.exists(dst):
            if not seconds or not truncate_video(src, dst, seconds):
                skipped += 1
                continue

        samples.append({
            "id": f"future_omni:{r['qid']}",
            "question": r.get("question", ""),
            "choices": r.get("options", r.get("choices", [])),
            "answer": r.get("answer", ""),
            "video_path": f"media/video/future_omni/{dst_name}",
            "task_type": r.get("task_type", "Forecasting"),
            "category": r.get("video_domain", ""),
            "audio_type": r.get("audio_type", ""),
            "meta": {
                "premise_seconds": seconds,
                "forecasting_pattern": r.get("forecasting_pattern", ""),
                "original_video": vid_name,
            },
        })

    data_path = os.path.join(out_dir, "data", "future_omni.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\n完成: {len(samples)} samples → {data_path}")
    print(f"视频:    {video_dst_dir}/  ({len(os.listdir(video_dst_dir))} files)")
    if skipped:
        print(f"跳过:    {skipped}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert OmniBench → unified JSON format.

Usage:
    python script/convert_omnibench.py \
        --jsonl-file /path/to/batch-N.jsonl \
        --mm-root /path/to/mm_data \
        --out-dir /path/to/output

Output:
    {out_dir}/data/omnibench.json
    {out_dir}/media/image/omnibench/{image_name}.{ext}
    {out_dir}/media/audio/omnibench/{audio_name}.{ext}
"""
import argparse, json, os, re, shutil

_OPT_RE = re.compile(r"(?P<L>[A-D])\s*[\.\)]\s*(?P<T>.+?)(?=\s+[A-D]\s*[\.\)]|$)", re.DOTALL)


def parse_options(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw]
    s = str(raw)
    m = _OPT_RE.findall(s)
    if m:
        return [f"{lt}. {tx.strip()}" for lt, tx in m]
    return [p.strip() for p in re.split(r"[\n;]", s) if p.strip()] or [s]


def text_to_letter(answer_text, options):
    """OmniBench stores full text of the correct option — reverse-match to A/B/C/D."""
    if not answer_text or not options:
        return (answer_text or "").strip().upper()[:1] or None
    letters, t = list("ABCDEFGHIJ"), answer_text.strip()
    for i, opt in enumerate(options):
        o = opt.strip()
        if o.startswith(letters[i] + "."):
            o = o[2:].strip()
        if o == t:
            return letters[i]
    return t.upper()[:1]


def main():
    p = argparse.ArgumentParser(description="Convert OmniBench to unified format")
    p.add_argument("--jsonl-file", required=True, help="Path to batch-N.jsonl")
    p.add_argument("--mm-root", required=True, help="Path to mm_data/ directory")
    p.add_argument("--out-dir", required=True, help="Output root directory")
    args = p.parse_args()

    data_dir = os.path.join(args.out_dir, "data")
    image_out = os.path.join(args.out_dir, "media", "image", "omnibench")
    audio_out = os.path.join(args.out_dir, "media", "audio", "omnibench")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(audio_out, exist_ok=True)

    records = []
    missing_img = missing_aud = 0

    with open(args.jsonl_file, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    total = len(lines)
    for i, line in enumerate(lines):
            r = json.loads(line)
            if (i + 1) % 20 == 0 or i + 1 == total:
                print(f"[{i + 1}/{total}] missing_img={missing_img} missing_aud={missing_aud}",
                      flush=True)

            options = parse_options(r.get("options") or r.get("option"))
            answer = text_to_letter(r.get("answer") or r.get("correct answer", ""), options)

            ip = ap = None

            img = r.get("image_path") or r.get("image")
            if img:
                src = os.path.join(args.mm_root, "image", img)
                dst = os.path.join(image_out, img)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    ip = f"media/image/omnibench/{img}"
                else:
                    missing_img += 1

            aud = r.get("audio_path") or r.get("audio")
            if aud:
                src = os.path.join(args.mm_root, "audio", aud)
                dst = os.path.join(audio_out, aud)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    ap = f"media/audio/omnibench/{aud}"
                else:
                    missing_aud += 1

            records.append({
                "id": f"omnibench:{r.get('index', '?')}",
                "question": r.get("question", ""),
                "choices": options,
                "answer": answer,
                "video_path": None,
                "audio_path": ap,
                "image_path": ip,
                "task_type": r.get("task type"),
                "category": None,
                "duration": None,
                "meta": {
                    "audio_type": r.get("audio type"),
                    "audio_content": r.get("audio content"),
                    "image_content": r.get("image content"),
                },
            })

    out_json = os.path.join(data_dir, "omnibench.json")
    json.dump(records, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"Done: {out_json} ({len(records)} records)")
    print(f"  image: {len(records) - missing_img}/{len(records)} copied")
    print(f"  audio: {len(records) - missing_aud}/{len(records)} copied")


if __name__ == "__main__":
    main()

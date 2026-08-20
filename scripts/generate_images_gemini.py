#!/usr/bin/env python3
"""Generate the same prompt manifest through Gemini, to compare against gpt-image-2.

The open defect in this pipeline is pattern consistency: repeating fills in the
real plans score ~11,800 on `scripts/pool_style_stats.py`, the delivered
generated pool 2,751, and round 1 through gpt-image-2 3,122. No prompt wording
has moved it, so the question is whether a different image model holds a ruled
fill better. This runs the identical prompts so the answer is a number.

    set -a; . ~/.env; set +a          # GEMINI_API_KEY
    python3 scripts/generate_images_gemini.py --ids 6,11,19,42 \
        --out data/image_generation/gemini_probe --model gemini-3-pro-image

Two differences from the OpenAI path, both handled here:

* Gemini takes a fixed aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9)
  rather than an arbitrary WIDTHxHEIGHT, and 21:9 is only 2.33:1 -- narrower
  than the eval set's median of 2.59. So a manifest row is generated at the
  widest allowed ratio not exceeding its target and then trimmed to width, the
  same way the 3:1 cap is handled for gpt-image-2.
* Size is `1K`/`2K`/`4K` rather than pixels. 2K is the closest match to the
  ~4 megapixels round 1 used; 4K is nearer the real pool's 5.9 median.

`--batch submit|status|fetch` uses Gemini's Batch API: half the token price, a
24-hour target window, and the job outlives the shell that submitted it. The
requests go up as a JSONL file rather than inline, because inline batches cap at
20MB and fifty 2K images come back far larger than that.
"""

import argparse
import base64
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Reuse the OpenAI path's manifest handling, crop and receipt rules so the two
# pools differ by the image model and nothing else.
_spec = importlib.util.spec_from_file_location(
    "gen_openai", Path(__file__).with_name("generate_images_openai.py"))
_openai = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_openai)
trim_to_aspect = _openai.trim_to_aspect
receipt_path = _openai.receipt_path
load_done = _openai.load_done
write_contact_sheet = _openai.write_contact_sheet

RATIOS = {"1:1": 1.0, "2:3": 2 / 3, "3:2": 1.5, "3:4": 0.75, "4:3": 4 / 3,
          "9:16": 9 / 16, "16:9": 16 / 9, "21:9": 21 / 9}


def choose_ratio(target):
    """Widest allowed ratio not exceeding the target; trimming does the rest."""
    below = [(value, name) for name, value in RATIOS.items() if value <= target]
    if not below:
        return min(RATIOS.items(), key=lambda kv: kv[1])[0]
    return max(below)[1]


def generate_one(client, types, row, out_dir, model, image_size, attempts,
                 default_ratio):
    dest = Path(out_dir) / row["filename"]
    target = row.get("aspect")
    ratio = choose_ratio(target) if target else default_ratio
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=model, contents=row["prompt"],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=ratio,
                                                   image_size=image_size)))
            blob = None
            for candidate in (response.candidates or []):
                for part in (candidate.content.parts or []):
                    data = getattr(part, "inline_data", None)
                    if data and data.data:
                        blob = data.data
                        break
                if blob:
                    break
            if not blob:
                raise RuntimeError(f"no image in response: "
                                   f"{getattr(response, 'text', '')[:200]}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(blob)
            tmp.replace(dest)
            cropped = (trim_to_aspect(dest, target)
                       if target and target > RATIOS[ratio] else None)
            usage = getattr(response, "usage_metadata", None)
            return {"id": row["id"], "filename": row["filename"],
                    "status": "accepted", "model": model, "size": image_size,
                    "aspect_ratio": ratio, "attempt": attempt,
                    "bytes": len(blob), "cropped_to": cropped,
                    "usage": (usage.model_dump() if hasattr(usage, "model_dump")
                              else None),
                    "timestamp": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:                     # noqa: BLE001 - reported below
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(min(2 ** attempt, 30))
    return {"id": row["id"], "filename": row["filename"], "status": "failed",
            "model": model, "size": image_size, "aspect_ratio": ratio,
            "attempt": attempts, "error": last_error,
            "timestamp": datetime.now(timezone.utc).isoformat()}



# --- Batch API ----------------------------------------------------------------

def batch_state_path(out_dir):
    return Path(out_dir) / "batch_state.json"


def build_batch_file(rows, out_dir, image_size, default_ratio):
    """One JSONL line per image, in the Gemini batch envelope."""
    path = Path(out_dir) / "batch_requests.jsonl"
    with path.open("w") as handle:
        for row in rows:
            ratio = choose_ratio(row["aspect"]) if row.get("aspect") else default_ratio
            handle.write(json.dumps({
                "key": f"item-{row['id']:03d}",
                "request": {
                    "contents": [{"parts": [{"text": row["prompt"]}], "role": "user"}],
                    "generation_config": {
                        "response_modalities": ["IMAGE"],
                        "image_config": {"aspect_ratio": ratio,
                                         "image_size": image_size}}}}) + "\n")
    return path


def batch_submit(client, rows, out_dir, model, image_size, default_ratio):
    path = build_batch_file(rows, out_dir, image_size, default_ratio)
    uploaded = client.files.upload(file=str(path),
                                   config={"mime_type": "application/jsonl"})
    job = client.batches.create(model=model, src=uploaded.name,
                                config={"display_name": Path(out_dir).name})
    state = {"job": job.name, "input_file": uploaded.name, "requests": len(rows),
             "model": model, "image_size": image_size,
             "submitted": datetime.now(timezone.utc).isoformat()}
    batch_state_path(out_dir).write_text(json.dumps(state, indent=1))
    print(f"submitted {len(rows)} requests as {job.name} (state {job.state})")
    print(f"check with: --batch status --out {out_dir}")
    return 0


def resolve_job(out_dir, job_name):
    if job_name:
        return job_name
    path = batch_state_path(out_dir)
    if not path.exists():
        raise SystemExit(f"no job name given and no {path}")
    return json.loads(path.read_text())["job"]


def batch_status(client, out_dir, job_name):
    job = client.batches.get(name=resolve_job(out_dir, job_name))
    print(f"{job.name}  state={job.state}")
    if str(job.state).endswith("SUCCEEDED"):
        print(f"fetch with: --batch fetch --out {out_dir}")
    return 0


def batch_fetch(client, rows, out_dir, job_name):
    """Write every returned image, crop it, and append its receipt."""
    job = client.batches.get(name=resolve_job(out_dir, job_name))
    if not str(job.state).endswith("SUCCEEDED"):
        print(f"job is {job.state}, not finished")
        return 1
    by_key = {f"item-{row['id']:03d}": row for row in rows}
    payload = client.files.download(file=job.dest.file_name).decode("utf-8")
    written = failed = 0
    with receipt_path(out_dir).open("a") as log:
        for line in payload.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            row = by_key.get(record.get("key"))
            if row is None:
                continue
            blob = None
            for candidate in ((record.get("response") or {}).get("candidates") or []):
                for part in (candidate.get("content") or {}).get("parts", []):
                    data = part.get("inlineData") or part.get("inline_data")
                    if data and data.get("data"):
                        blob = base64.b64decode(data["data"])
                        break
                if blob:
                    break
            if blob is None:
                failed += 1
                receipt = {"id": row["id"], "filename": row["filename"],
                           "status": "failed",
                           "error": str(record.get("error") or record)[:400],
                           "timestamp": datetime.now(timezone.utc).isoformat()}
            else:
                dest = Path(out_dir) / row["filename"]
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(blob)
                tmp.replace(dest)
                target = row.get("aspect")
                cropped = trim_to_aspect(dest, target) if target else None
                written += 1
                receipt = {"id": row["id"], "filename": row["filename"],
                           "status": "accepted", "model": job.model, "via": "batch",
                           "bytes": len(blob), "cropped_to": cropped,
                           "timestamp": datetime.now(timezone.utc).isoformat()}
            log.write(json.dumps(receipt) + "\n")
            log.flush()
    print(f"wrote {written} images, {failed} failed -> {out_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default="image_generation/prompts_v4.jsonl")
    ap.add_argument("--out", default="data/image_generation/gemini_probe")
    ap.add_argument("--model", default="gemini-3-pro-image",
                    help="gemini-3-pro-image, gemini-3.1-flash-image, ...")
    ap.add_argument("--image-size", default="2K", choices=["1K", "2K", "4K"])
    ap.add_argument("--aspect", default="16:9", choices=sorted(RATIOS),
                    help="fallback when a manifest row carries no aspect")
    ap.add_argument("--ids", help="comma-separated manifest IDs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--report", help="write a contact sheet here when done")
    ap.add_argument("--batch", choices=["submit", "status", "fetch"],
                    help="use Gemini's Batch API: half price, 24h window")
    ap.add_argument("--batch-id", help="override the job name in the state file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.prompts).read_text().splitlines()
            if line.strip()]
    if args.ids:
        wanted = {int(v) for v in args.ids.split(",") if v.strip()}
        rows = [r for r in rows if r["id"] in wanted]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(out_dir)
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]

    if args.batch in {"status", "fetch"}:
        if not os.environ.get("GEMINI_API_KEY"):
            print("GEMINI_API_KEY is not set", file=sys.stderr)
            return 2
        from google import genai
        client = genai.Client()
        if args.batch == "status":
            return batch_status(client, out_dir, args.batch_id)
        return batch_fetch(client, rows, out_dir, args.batch_id)
    print(f"manifest={len(rows)}  already accepted={len(done)}  to generate={len(todo)}")
    print(f"model={args.model} size={args.image_size}")
    if args.dry_run:
        for r in todo:
            ratio = choose_ratio(r["aspect"]) if r.get("aspect") else args.aspect
            note = (f" -> crop to {r['aspect']}:1"
                    if r.get("aspect", 0) > RATIOS[ratio] else "")
            print(f"  would generate {r['id']:>3} {r['filename']} {ratio}{note}")
        return 0
    if not todo:
        print("nothing to do")
        return 0
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set (try: set -a; . ~/.env; set +a)",
              file=sys.stderr)
        return 2

    from google import genai
    from google.genai import types
    client = genai.Client()

    if args.batch == "submit":
        return batch_submit(client, todo, out_dir, args.model, args.image_size,
                            args.aspect)

    receipts, failures = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            receipt_path(out_dir).open("a") as log:
        futures = {pool.submit(generate_one, client, types, r, out_dir, args.model,
                               args.image_size, args.attempts, args.aspect): r
                   for r in todo}
        for n, future in enumerate(as_completed(futures), 1):
            receipt = future.result()
            log.write(json.dumps(receipt) + "\n")
            log.flush()
            receipts.append(receipt)
            failures += receipt["status"] == "failed"
            mark = "ok " if receipt["status"] == "accepted" else "FAIL"
            note = "" if receipt["status"] == "accepted" else f"  {receipt['error']}"
            print(f"[{n}/{len(todo)}] {mark} {receipt['filename']}{note}", flush=True)

    accepted = [r for r in receipts if r["status"] == "accepted"]
    print(f"\naccepted={len(accepted)} failed={failures} -> {out_dir}")
    if args.report and accepted:
        path = write_contact_sheet(out_dir, accepted, args.report)
        if path:
            print(f"contact sheet: {path}")
    return 1 if failures and not accepted else 0


if __name__ == "__main__":
    raise SystemExit(main())

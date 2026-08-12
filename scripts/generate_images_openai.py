#!/usr/bin/env python3
"""Generate the unlabelled realistic-plan pool with an OpenAI image model.

Source count is the binding constraint on this project: 28 -> 86 real sources
bought +0.073 on HF14, one generated plan is worth about as much as one real
plan, and re-augmenting the same 114 sources 18x -> 72x buys nothing. So the way
to move the number is more DISTINCT plans, and generation is the only supply
that can also choose its own resolution (worth ~0.032, and unretrofittable onto
the natively-640px scraped pool).

Reads the prompt manifest rendered by `render_image_generation_prompts.py`
(use `--version v2`: the v1 framing asked for whole sheets and yielded 29%).
Writes each image to its manifest `filename` and appends a receipt row, so a run
is resumable and an accepted image is never silently overwritten.

    export OPENAI_API_KEY=...        # or `set -a; . ~/.env; set +a`
    python3 scripts/generate_images_openai.py \
        --prompts image_generation/prompts_v2.jsonl \
        --out data/image_generation/realistic_label_pool_v2 --limit 4

Nothing here judges the drawings. Generation is stochastic and the failure modes
that mattered last round (legends, title blocks, cropped fragments, tonal
shading) are visual, so the images still need looking at before labelling --
`--report` writes a contact sheet to make that quick.
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# gpt-image-1's landscape option is 1536x1024. The README asks for >=1500px on
# the long side because training runs at 1280 and downscaling the 640px scraped
# plans is what costs ~0.032, so 1536 clears the bar with nothing to spare.
SIZES = {"landscape": "1536x1024", "portrait": "1024x1536", "square": "1024x1024"}


def receipt_path(out_dir):
    return Path(out_dir) / "generation_receipt.jsonl"


def load_done(out_dir):
    """IDs with BOTH an accepted receipt row and a file on disk."""
    path = receipt_path(out_dir)
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "accepted" and (Path(out_dir) / row["filename"]).exists():
            done.add(row["id"])
    return done


def generate_one(client, row, out_dir, model, size, quality, attempts):
    """Return a receipt dict. Retries transient API failures with backoff."""
    dest = Path(out_dir) / row["filename"]
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            result = client.images.generate(model=model, prompt=row["prompt"],
                                            size=size, quality=quality, n=1)
            payload = result.data[0]
            if getattr(payload, "b64_json", None):
                blob = base64.b64decode(payload.b64_json)
            else:                                    # some models return a URL
                import urllib.request
                with urllib.request.urlopen(payload.url) as response:
                    blob = response.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(blob)
            tmp.replace(dest)                        # never a half-written accept
            return {"id": row["id"], "filename": row["filename"],
                    "status": "accepted", "model": model, "size": size,
                    "quality": quality, "attempt": attempt,
                    "bytes": len(blob),
                    "timestamp": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:                     # noqa: BLE001 - reported below
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(min(2 ** attempt, 30))
    return {"id": row["id"], "filename": row["filename"], "status": "failed",
            "model": model, "size": size, "quality": quality,
            "attempt": attempts, "error": last_error,
            "timestamp": datetime.now(timezone.utc).isoformat()}


def write_contact_sheet(out_dir, rows, path, cols=4, thumb=420):
    """One page of thumbnails so the visual gates can be checked at a glance."""
    from PIL import Image, ImageDraw
    files = [Path(out_dir) / r["filename"] for r in rows
             if (Path(out_dir) / r["filename"]).exists()]
    if not files:
        return None
    cols = min(cols, len(files))
    rows_n = (len(files) + cols - 1) // cols
    pad, label = 8, 18
    sheet = Image.new("RGB", (cols * (thumb + pad) + pad,
                              rows_n * (thumb + pad + label) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        im.thumbnail((thumb, thumb))
        x = pad + (i % cols) * (thumb + pad)
        y = pad + (i // cols) * (thumb + pad + label)
        sheet.paste(im, (x, y))
        draw.text((x, y + im.height + 4), f"{f.name}  {im.width}x{im.height}",
                  fill="black")
    sheet.save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default="image_generation/prompts_v2.jsonl")
    ap.add_argument("--out", default="data/image_generation/realistic_label_pool_v2")
    ap.add_argument("--model", default="gpt-image-1",
                    help="image model id; override if a newer one is available")
    ap.add_argument("--orientation", choices=sorted(SIZES), default="landscape")
    ap.add_argument("--quality", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--ids", help="comma-separated manifest IDs (default: all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many NEW images (0 = no limit)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--report", help="write a contact sheet here when done")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be generated and exit")
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

    size = SIZES[args.orientation]
    print(f"manifest={len(rows)}  already accepted={len(done)}  to generate={len(todo)}")
    print(f"model={args.model} size={size} quality={args.quality}")
    if args.dry_run:
        for r in todo:
            print(f"  would generate {r['id']:>3} {r['filename']}")
        return 0
    if not todo:
        print("nothing to do")
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (try: set -a; . ~/.env; set +a)",
              file=sys.stderr)
        return 2

    from openai import OpenAI
    client = OpenAI()

    receipts, failures = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            receipt_path(out_dir).open("a") as log:
        futures = {pool.submit(generate_one, client, r, out_dir, args.model,
                               size, args.quality, args.attempts): r for r in todo}
        for n, future in enumerate(as_completed(futures), 1):
            receipt = future.result()
            log.write(json.dumps(receipt) + "\n")
            log.flush()                              # resumable if interrupted
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
    print("\nNext: look at every image before labelling. Reject legends, title\n"
          "blocks, cropped fragments and tonal shading -- last round's yield was\n"
          "28/98. Then upload survivors with scripts/upload_unlabelled_roboflow.py.")
    return 1 if failures and not accepted else 0


if __name__ == "__main__":
    raise SystemExit(main())

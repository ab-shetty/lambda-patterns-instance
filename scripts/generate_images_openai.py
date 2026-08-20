#!/usr/bin/env python3
"""Generate the unlabelled realistic-plan pool with an OpenAI image model.

Source count is the binding constraint on this project: 28 -> 86 real sources
bought +0.073 on HF14, one generated plan is worth about as much as one real
plan, and re-augmenting the same 114 sources 18x -> 72x buys nothing. So the way
to move the number is more DISTINCT plans, and generation is the only supply
that can also choose its own resolution (worth ~0.032, and unretrofittable onto
the natively-640px scraped pool).

Reads the prompt manifest rendered by `render_image_generation_prompts.py`
(use `--version v4`, which is aimed at the evaluation set's measured shape; v1-v3
are earlier framings kept so their results stay reproducible). Writes each image
to its manifest `filename` and appends a receipt row, so a run is resumable and
an accepted image is never silently overwritten.

For a full round use `--batch submit`: the Batch API halves the token price
($15 per 1M image output tokens against $30), which takes a 300-image round from
~$60 to ~$30, in exchange for a 24-hour completion window. Submit, then fetch
later -- the batch outlives this shell, and the state file in the output
directory holds the id.

Two things the manifest drives per image. A v4 row carries the aspect ratio it
wants, and the eval set runs to 5:1 while the API stops at 3:1 -- so anything
wider is generated at 3:1 and trimmed to width by dropping the emptiest rows,
which is what a page excerpt is anyway. And `--min-regularity` re-rolls an image
whose material fills are not periodic enough, measured the same way as
`scripts/pool_style_stats.py`: the real plans read ~18,600, the delivered
generated pool ~3,200. No prompt wording has closed that gap, so the lever is
selection.

    export OPENAI_API_KEY=...        # or `set -a; . ~/.env; set +a`
    python3 scripts/generate_images_openai.py \
        --prompts image_generation/prompts_v3.jsonl \
        --out data/image_generation/realistic_label_pool_v3 --limit 4

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

# gpt-image-1 offered three fixed sizes and its landscape option, 1536x1024,
# cleared the README's >=1500px aim with nothing to spare. gpt-image-2 takes an
# arbitrary WIDTHxHEIGHT instead, so resolution is now a real choice: both axes
# divisible by 16, aspect ratio within 1:3..3:1, up to 3840x2160. Training runs
# at 1280 and downscaling the 640px scraped plans is what costs ~0.032, so
# generate well above 1280 -- the hatch spacing has to survive the downscale, and
# resolution can never be retrofitted onto a pool once it exists.
SIZES = {"landscape": "2496x1664",       # 3:2, the sheet-like default
         "wide": "2560x1440",            # 16:9
         "portrait": "1664x2496",
         "square": "2048x2048",
         "legacy-landscape": "1536x1024"}  # what gpt-image-1 could do
MAX_PIXELS = (3840, 2160)
MAX_ASPECT = 3.0                          # the API's limit, not a choice


def check_size(size):
    """Reject a size the API would reject, before spending a request on it."""
    try:
        width, height = (int(v) for v in size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--size must look like 2496x1664, got {size!r}")
    if width % 16 or height % 16:
        raise SystemExit(f"--size {size}: both axes must be divisible by 16")
    if not 1 / 3 <= width / height <= 3:
        raise SystemExit(f"--size {size}: aspect ratio must be within 1:3..3:1")
    if width * height > MAX_PIXELS[0] * MAX_PIXELS[1]:
        raise SystemExit(f"--size {size}: above the "
                         f"{MAX_PIXELS[0]}x{MAX_PIXELS[1]} maximum")
    if min(width, height) < 384:
        print(f"warning: {size} has a short side under 384; the upload gate in "
              f"scripts/upload_unlabelled_roboflow.py rejects those")
    return size


def size_for_aspect(aspect, budget=4_150_000):
    """Largest API-legal size at (or nearest to) this aspect, under a budget.

    Billing is per output token and tokens track pixels, so the budget is what
    keeps a wide image from costing more than the 2496x1664 default. Aspects
    past 3:1 are generated at 3:1 and trimmed afterwards.
    """
    target = min(aspect, MAX_ASPECT)
    height = int((budget / target) ** 0.5) // 16 * 16
    height = max(16, min(height, MAX_PIXELS[1]))
    width = int(height * target) // 16 * 16
    if width > MAX_PIXELS[0]:
        width = MAX_PIXELS[0] // 16 * 16
        height = int(width / target) // 16 * 16
    return f"{width}x{height}"


def trim_to_aspect(path, aspect):
    """Crop toward a wider aspect WITHOUT cutting the drawing.

    The trim exists because the APIs cap aspect (3:1 on gpt-image-2, 21:9 on
    Gemini) below the eval set's tail. It used to take the densest band of the
    requested height, which on a 21:9 generation asked to reach 4.6:1 meant
    slicing the bottom off the building. So the ink decides: rows carrying
    drawing are never removed, and if reaching the target would cut them, the
    crop stops at the widest aspect the margins allow and the receipt records
    what was actually achieved.
    """
    from PIL import Image
    import numpy as np
    with Image.open(path) as im:
        im = im.convert("RGB")
        width, height = im.size
        wanted = int(round(width / aspect))
        if wanted >= height:
            return None
        gray = np.asarray(im.convert("L"), dtype=np.float32)
        page = float(np.percentile(gray, 95))
        row_ink = (gray < 0.92 * max(page, 1e-6)).mean(axis=1)
        drawn = np.flatnonzero(row_ink > 0.002)          # rows holding anything
        if drawn.size == 0:
            return None
        pad = max(8, int(0.01 * height))
        top_limit = max(0, int(drawn[0]) - pad)
        bottom_limit = min(height, int(drawn[-1]) + pad)
        keep = max(wanted, bottom_limit - top_limit)     # never cut the drawing
        if keep >= height:
            return None
        centre = (top_limit + bottom_limit) / 2
        top = int(min(max(0, centre - keep / 2), height - keep))
        im.crop((0, top, width, top + keep)).save(path)
    return f"{width}x{keep}"


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


def is_retryable(exc):
    """Rate limits, timeouts and 5xx are worth another attempt; a 400 is not.

    A rejected size or a refused prompt fails identically three times, so
    retrying one only wastes wall-clock on a run of 300.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        return True                                  # connection/timeout/unknown
    return status == 429 or status >= 500


def regularity_of(path):
    """Median fill periodicity, the same measure as scripts/pool_style_stats.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pool_style_stats", Path(__file__).with_name("pool_style_stats.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.image_stats(Path(path))["regularity"]


def generate_one(client, row, out_dir, model, size, quality, attempts,
                 output_format="png", min_regularity=0):
    """Return a receipt dict. Retries transient API failures with backoff."""
    # the manifest names every file .png; honour --output-format instead of
    # writing a jpeg under a .png name, and record what was actually written
    ext = "jpg" if output_format == "jpeg" else output_format
    filename = Path(row["filename"]).with_suffix("." + ext).name
    dest = Path(out_dir) / filename
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            result = client.images.generate(model=model, prompt=row["prompt"],
                                            size=size, quality=quality,
                                            output_format=output_format, n=1)
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
            cropped = None
            if row.get("aspect") and row["aspect"] > MAX_ASPECT:
                cropped = trim_to_aspect(dest, row["aspect"])
            regularity = regularity_of(dest) if min_regularity else None
            # `None` means the image held no measurable fill, which is not the
            # same as a wobbly one -- do not re-roll for it.
            if min_regularity and regularity is not None and regularity < min_regularity:
                last_error = (f"regularity {regularity} below {min_regularity}")
                if attempt < attempts:
                    continue                         # re-roll: fills too wobbly
                break
            usage = getattr(result, "usage", None)
            return {"id": row["id"], "filename": filename,
                    "status": "accepted", "model": model, "size": size,
                    "quality": quality, "attempt": attempt,
                    "bytes": len(blob), "cropped_to": cropped,
                    "regularity": regularity,
                    "usage": usage.model_dump() if usage else None,
                    "timestamp": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:                     # noqa: BLE001 - reported below
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts and is_retryable(exc):
                time.sleep(min(2 ** attempt, 30))
                continue
            break
    return {"id": row["id"], "filename": filename, "status": "failed",
            "model": model, "size": size, "quality": quality,
            "attempt": attempt, "error": last_error,
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



# --- Batch API ----------------------------------------------------------------
# Half price, 24h window, and it survives the session that submitted it. The
# output file carries every image as base64, so it is large: budget ~5.5MB per
# 4-megapixel PNG, i.e. a couple of GB for a 300-image round.

def batch_state_path(out_dir):
    return Path(out_dir) / "batch_state.json"


def build_batch_file(rows, out_dir, model, quality, output_format, per_spec_aspect,
                     size):
    """One JSONL line per image, in the Batch API's envelope format."""
    path = Path(out_dir) / "batch_requests.jsonl"
    with path.open("w") as handle:
        for row in rows:
            body = {"model": model, "prompt": row["prompt"], "n": 1,
                    "quality": quality, "output_format": output_format,
                    "size": (size_for_aspect(row["aspect"])
                             if per_spec_aspect and row.get("aspect") else size)}
            handle.write(json.dumps({"custom_id": f"item-{row['id']:03d}",
                                     "method": "POST",
                                     "url": "/v1/images/generations",
                                     "body": body}) + "\n")
    return path


def batch_submit(client, rows, out_dir, args, size):
    path = build_batch_file(rows, out_dir, args.model, args.quality,
                            args.output_format, args.per_spec_aspect, size)
    upload = client.files.create(file=path.open("rb"), purpose="batch")
    batch = client.batches.create(input_file_id=upload.id,
                                  endpoint="/v1/images/generations",
                                  completion_window="24h",
                                  metadata={"round": Path(out_dir).name})
    state = {"batch_id": batch.id, "input_file_id": upload.id,
             "requests": len(rows), "model": args.model, "quality": args.quality,
             "submitted": datetime.now(timezone.utc).isoformat()}
    batch_state_path(out_dir).write_text(json.dumps(state, indent=1))
    print(f"submitted {len(rows)} requests as {batch.id} (status {batch.status})")
    print(f"state: {batch_state_path(out_dir)}")
    print(f"check with: --batch status --out {out_dir}")
    return 0


def resolve_batch_id(out_dir, batch_id):
    if batch_id:
        return batch_id
    path = batch_state_path(out_dir)
    if not path.exists():
        raise SystemExit(f"no batch id given and no {path}")
    return json.loads(path.read_text())["batch_id"]


def batch_status(client, out_dir, batch_id):
    batch = client.batches.retrieve(resolve_batch_id(out_dir, batch_id))
    counts = batch.request_counts
    print(f"{batch.id}  status={batch.status}  "
          f"completed={counts.completed}/{counts.total}  failed={counts.failed}")
    if batch.status == "completed":
        print(f"fetch with: --batch fetch --out {out_dir}")
    return 0 if batch.status in {"completed", "in_progress", "validating",
                                 "finalizing"} else 1


def batch_fetch(client, rows, out_dir, batch_id, output_format):
    """Write every completed image, crop it, and append its receipt."""
    batch = client.batches.retrieve(resolve_batch_id(out_dir, batch_id))
    if batch.status != "completed":
        print(f"batch is {batch.status}, not completed")
        return 1
    by_id = {f"item-{row['id']:03d}": row for row in rows}
    payload = client.files.content(batch.output_file_id).text
    written = failed = 0
    with receipt_path(out_dir).open("a") as log:
        for line in payload.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            row = by_id.get(record["custom_id"])
            if row is None:
                continue
            response = record.get("response") or {}
            body = response.get("body") or {}
            if response.get("status_code") != 200 or not body.get("data"):
                failed += 1
                receipt = {"id": row["id"], "filename": row["filename"],
                           "status": "failed", "error": str(record.get("error") or
                                                            body)[:400],
                           "timestamp": datetime.now(timezone.utc).isoformat()}
            else:
                ext = "jpg" if output_format == "jpeg" else output_format
                filename = Path(row["filename"]).with_suffix("." + ext).name
                dest = Path(out_dir) / filename
                blob = base64.b64decode(body["data"][0]["b64_json"])
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(blob)
                tmp.replace(dest)
                cropped = (trim_to_aspect(dest, row["aspect"])
                           if row.get("aspect", 0) > MAX_ASPECT else None)
                written += 1
                receipt = {"id": row["id"], "filename": filename,
                           "status": "accepted", "model": body.get("model"),
                           "size": body.get("size"), "via": "batch",
                           "bytes": len(blob), "cropped_to": cropped,
                           "usage": body.get("usage"),
                           "timestamp": datetime.now(timezone.utc).isoformat()}
            log.write(json.dumps(receipt) + "\n")
            log.flush()
    print(f"wrote {written} images, {failed} failed -> {out_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default="image_generation/prompts_v4.jsonl")
    ap.add_argument("--out", default="data/image_generation/realistic_label_pool_v4")
    ap.add_argument("--model", default="gpt-image-2",
                    help="image model id; override if a newer one is available")
    ap.add_argument("--orientation", choices=sorted(SIZES), default="landscape",
                    help="named preset; ignored when --size is given")
    ap.add_argument("--size", help="explicit WIDTHxHEIGHT, e.g. 2496x1664 "
                                   "(gpt-image-2 only; axes divisible by 16)")
    ap.add_argument("--output-format", default="png", choices=["png", "jpeg", "webp"])
    ap.add_argument("--per-spec-aspect", action="store_true", default=True,
                    help="take each image's aspect from the manifest (v4 rows)")
    ap.add_argument("--fixed-aspect", dest="per_spec_aspect", action="store_false",
                    help="use one --size/--orientation for every image")
    ap.add_argument("--min-regularity", type=int, default=0,
                    help="re-roll images whose fills score below this "
                         "(real plans ~18600, generated pool ~3200; 0 = off)")
    ap.add_argument("--quality", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--ids", help="comma-separated manifest IDs (default: all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many NEW images (0 = no limit)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--report", help="write a contact sheet here when done")
    ap.add_argument("--batch", choices=["submit", "status", "fetch"],
                    help="use the Batch API: half price, 24h window. submit, "
                         "then status, then fetch")
    ap.add_argument("--batch-id", help="override the id in the state file")
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

    size = check_size(args.size or SIZES[args.orientation])
    if args.batch in {"status", "fetch"}:                # nothing to plan here
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set", file=sys.stderr)
            return 2
        client = OpenAI()
        if args.batch == "status":
            return batch_status(client, out_dir, args.batch_id)
        return batch_fetch(client, rows, out_dir, args.batch_id, args.output_format)
    print(f"manifest={len(rows)}  already accepted={len(done)}  to generate={len(todo)}")
    per_spec = args.per_spec_aspect and any(r.get("aspect") for r in todo)
    print(f"model={args.model} size={'per-spec' if per_spec else size} "
          f"quality={args.quality} format={args.output_format} "
          f"min-regularity={args.min_regularity or 'off'}")
    if args.dry_run:
        for r in todo:
            planned = (size_for_aspect(r["aspect"])
                       if args.per_spec_aspect and r.get("aspect") else size)
            note = (f" -> crop to {r['aspect']}:1" if r.get("aspect", 0) > MAX_ASPECT
                    else "")
            print(f"  would generate {r['id']:>3} {r['filename']} "
                  f"{planned}{note}  {r.get('category', '')}")
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

    if args.batch == "submit":
        return batch_submit(client, todo, out_dir, args, size)

    receipts, failures = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            receipt_path(out_dir).open("a") as log:
        futures = {pool.submit(generate_one, client, r, out_dir, args.model,
                               size_for_aspect(r["aspect"])
                               if args.per_spec_aspect and r.get("aspect") else size,
                               args.quality, args.attempts, args.output_format,
                               args.min_regularity): r for r in todo}
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
    billed = [r["usage"] for r in accepted if r.get("usage")]
    if billed:
        print(f"tokens: input={sum(u.get('input_tokens', 0) for u in billed)} "
              f"output={sum(u.get('output_tokens', 0) for u in billed)} "
              f"over {len(billed)} images")
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

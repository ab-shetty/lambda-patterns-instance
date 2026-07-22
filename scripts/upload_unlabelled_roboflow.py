#!/usr/bin/env python3
"""Upload the checked 100-image generation batch with a resumable receipt."""

import argparse
import json
import os
from pathlib import Path

from roboflow import Roboflow
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--workspace", default="perceive-ai")
    parser.add_argument("--project", default="floz-real-pool")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--receipt", default=None)
    parser.add_argument("--execute", action="store_true",
                        help="Perform uploads; without this flag only validate/plan.")
    args = parser.parse_args()
    root = Path(args.images)
    expected = [root / f"floz_gen_{item:03d}.png" for item in range(1, 101)]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise SystemExit(f"refusing partial batch: {len(missing)} images missing; "
                         f"first missing: {missing[0]}")
    for path in expected:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
        except Exception as error:
            raise SystemExit(f"invalid image {path}: {error}") from error
        if width < 1024 or height < 1024 or width <= height:
            raise SystemExit(
                f"quality gate failed for {path}: expected landscape with both "
                f"axes >=1024, got {width}x{height}")
    receipt = Path(args.receipt or root / "roboflow_upload_receipt.jsonl")
    completed = set()
    if receipt.exists():
        completed = {json.loads(line)["filename"]
                     for line in receipt.read_text().splitlines() if line.strip()}
    pending = [path for path in expected if path.name not in completed]
    print(json.dumps({"images": len(expected), "already_uploaded": len(completed),
                      "pending": len(pending), "batch": args.batch,
                      "execute": args.execute}, indent=2))
    if not args.execute:
        return
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY is not set")
    project = (Roboflow(api_key=api_key).workspace(args.workspace)
               .project(args.project))
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("a") as stream:
        for index, path in enumerate(pending, 1):
            project.upload(
                image_path=str(path), split="train", num_retry_uploads=3,
                batch_name=args.batch,
                tag_names=["generated-realistic-v1", "needs-labels"],
                metadata={"generation_manifest_id": int(path.stem[-3:]),
                          "annotation_status": "unlabelled"})
            row = {"filename": path.name, "batch": args.batch,
                   "workspace": args.workspace, "project": args.project}
            stream.write(json.dumps(row) + "\n"); stream.flush()
            print(f"uploaded {index}/{len(pending)}: {path.name}")


if __name__ == "__main__":
    main()

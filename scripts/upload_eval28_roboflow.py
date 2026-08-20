#!/usr/bin/env python3
"""Put the 28 real evaluation images into Roboflow WITH their annotations.

These are the HF `real-world-test` images -- the acceptance set. Having them in
the labelling tool alongside a generated round is how you see, rather than argue
about, whether a round looks like the thing the metric is scored on.

    python3 scripts/upload_eval28_roboflow.py --project floz-eval28-reference \
        --create --execute

EVALUATION-ONLY. `startup.md` is explicit that these images are never trained on.
Putting them in a Roboflow project makes it possible to export them into a
training mix by accident, so the project is named and tagged to make that hard to
do without noticing, and it must never be merged into a pool that feeds
`merge_local_datasets.py`. The 14 HF14 acceptance images are additionally tagged.

Hole convention: an annotation's `segmentation` is a list of rings, the first the
outer boundary and the rest holes. Roboflow carries holes as separate `remove`
polygons -- that is what `scripts/roboflow_to_local.py` reads back -- so this
writes the outer ring under its `pattern*` name and every hole as `remove`.
"""

import argparse
import io
import json
import os
from pathlib import Path

from PIL import Image

HF14 = [12, 16, 27, 7, 11, 25, 23, 1, 18, 2, 0, 3, 14, 24]


def coco_for_row(index, filename, width, height, annotations):
    """One image's COCO, with holes split out as `remove` polygons."""
    names, records = [], []
    for annotation in annotations:
        rings = annotation.get("segmentation") or []
        if not rings:
            continue
        for position, ring in enumerate(rings):
            if len(ring) < 6:
                continue
            name = annotation.get("category_name", "pattern1") if position == 0 else "remove"
            if name not in names:
                names.append(name)
            xs, ys = ring[0::2], ring[1::2]
            records.append({"name": name, "segmentation": [list(map(float, ring))],
                            "bbox": [min(xs), min(ys), max(xs) - min(xs),
                                     max(ys) - min(ys)],
                            "area": float(annotation.get("area", 0)) if position == 0 else 0.0})
    categories = [{"id": i + 1, "name": n, "supercategory": "pattern"}
                  for i, n in enumerate(names)]
    ids = {c["name"]: c["id"] for c in categories}
    return {"info": {"description": f"floz real-world-test index {index}"},
            "images": [{"id": 1, "file_name": filename, "width": width,
                        "height": height}],
            "categories": categories,
            "annotations": [{"id": i + 1, "image_id": 1,
                             "category_id": ids[r["name"]],
                             "segmentation": r["segmentation"], "bbox": r["bbox"],
                             "area": r["area"], "iscrowd": 0}
                            for i, r in enumerate(records)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default="perceive-ai")
    ap.add_argument("--project", default="floz-eval28-reference")
    ap.add_argument("--batch", default="real-world-test-28")
    ap.add_argument("--staging", default="data/eval28_roboflow")
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    rows = load_dataset("abshetty/floz-synth-v5", "real-world-test", split="test",
                        cache_dir="./data")
    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)

    prepared = []
    for index, row in enumerate(rows):
        raw = row["image"]
        if isinstance(raw, dict):
            raw = raw["bytes"]
        image = (Image.open(io.BytesIO(raw)) if isinstance(raw, (bytes, bytearray))
                 else raw).convert("RGB")
        name = f"eval_{index:02d}.png"
        image_path = staging / name
        if not image_path.exists():
            image.save(image_path)
        annotations = row["annotations"]
        if isinstance(annotations, str):
            annotations = json.loads(annotations)
        coco = coco_for_row(index, name, image.width, image.height, annotations)
        coco_path = staging / f"eval_{index:02d}.coco.json"
        coco_path.write_text(json.dumps(coco))
        prepared.append((index, image_path, coco_path, len(coco["annotations"]),
                         image.size))

    total = sum(p[3] for p in prepared)
    print(json.dumps({"images": len(prepared), "annotations": total,
                      "project": args.project, "execute": args.execute}, indent=2))
    if not args.execute:
        for index, image_path, _, count, size in prepared[:5]:
            print(f"  {index:>2} {image_path.name} {size[0]}x{size[1]} "
                  f"{count} polygons")
        print("  ... dry run; pass --execute to upload")
        return

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "uploader", Path(__file__).with_name("upload_unlabelled_roboflow.py"))
    uploader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uploader)
    workspace = uploader.get_workspace(args.workspace)
    project = uploader.open_project(workspace, args.project, args.create)

    receipt = staging / "roboflow_upload_receipt.jsonl"
    done = set()
    if receipt.exists():
        done = {json.loads(line)["filename"]
                for line in receipt.read_text().splitlines() if line.strip()}
    with receipt.open("a") as log:
        for index, image_path, coco_path, count, _ in prepared:
            if image_path.name in done:
                continue
            tags = ["real-world-test", "eval-only", "do-not-train"]
            if index in HF14:
                tags.append("hf14-acceptance")
            project.single_upload(
                image_path=str(image_path), annotation_path=str(coco_path),
                split="train", num_retry_uploads=3, batch_name=args.batch,
                tag_names=tags,
                metadata={"eval_index": index, "annotation_status": "annotated"})
            log.write(json.dumps({"filename": image_path.name, "index": index,
                                  "polygons": count}) + "\n")
            log.flush()
            print(f"uploaded eval {index:>2} ({count} polygons)", flush=True)


if __name__ == "__main__":
    main()

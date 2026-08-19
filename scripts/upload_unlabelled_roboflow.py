#!/usr/bin/env python3
"""Upload a generated image batch to Roboflow, unlabelled, with a resumable receipt.

Every image in `--images` is validated before anything is uploaded, so a batch
either goes up whole or not at all. `--expect N` additionally demands exactly the
manifest filenames `floz_gen_001..N.png`, which is how the 100-image v1 round was
uploaded; without it the directory contents are the batch, which is what a smoke
test needs.

    # validate only
    python3 scripts/upload_unlabelled_roboflow.py \
        --images data/image_generation/realistic_label_pool_v3 \
        --project floz-gen-v3-smoke --batch v3-smoke --create
    # then upload
    ... --execute

A review project created for a smoke test is disposable; remove it with
`--delete-project --execute`, which moves it to Roboflow's trash (recoverable
there for 30 days).
"""

import argparse
import json
import os
from pathlib import Path

from roboflow import Roboflow
from PIL import Image

SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def collect(root, expect):
    """The batch, validated. Raises rather than uploading a partial round."""
    if expect:
        paths = [root / f"floz_gen_{item:03d}.png" for item in range(1, expect + 1)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise SystemExit(f"refusing partial batch: {len(missing)} images missing; "
                             f"first missing: {missing[0]}")
    else:
        paths = sorted(p for p in root.iterdir() if p.suffix.lower() in SUFFIXES)
        if not paths:
            raise SystemExit(f"no images in {root}")
    for path in paths:
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
    return paths


def get_workspace(workspace):
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY is not set")
    return Roboflow(api_key=api_key).workspace(workspace)


def open_project(workspace, name, create, annotation="pattern"):
    """Existing project, or a new empty one when --create is given."""
    try:
        return workspace.project(name)
    except Exception:
        if not create:
            raise SystemExit(f"project {name!r} not found (pass --create to make it)")
    project = workspace.create_project(project_name=name,
                                       project_type="instance-segmentation",
                                       project_license="CC BY 4.0",
                                       annotation=annotation)
    print(f"created project {name!r} (instance-segmentation, annotation group "
          f"{annotation!r})")
    return project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images")
    parser.add_argument("--workspace", default="perceive-ai")
    parser.add_argument("--project", default="floz-real-pool")
    parser.add_argument("--batch")
    parser.add_argument("--expect", type=int, default=0,
                        help="require exactly floz_gen_001..N.png (v1 round used 100)")
    parser.add_argument("--tags", default="generated-realistic-v1,needs-labels")
    parser.add_argument("--annotation", default="pattern",
                        help="annotation group name when creating a project")
    parser.add_argument("--create", action="store_true",
                        help="create the project if it does not exist")
    parser.add_argument("--delete-project", action="store_true",
                        help="move the project to Roboflow's trash and exit")
    parser.add_argument("--receipt", default=None)
    parser.add_argument("--execute", action="store_true",
                        help="Perform uploads; without this flag only validate/plan.")
    args = parser.parse_args()

    if args.delete_project:
        print(json.dumps({"action": "delete_project", "workspace": args.workspace,
                          "project": args.project, "execute": args.execute}, indent=2))
        if not args.execute:
            print("dry run: pass --execute to move this project to the trash")
            return
        response = get_workspace(args.workspace).project(args.project).delete()
        print(f"deleted {args.project}: {response}")
        return

    if not args.images or not args.batch:
        raise SystemExit("--images and --batch are required unless --delete-project")
    root = Path(args.images)
    paths = collect(root, args.expect)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    receipt = Path(args.receipt or root / "roboflow_upload_receipt.jsonl")
    completed = set()
    if receipt.exists():
        completed = {json.loads(line)["filename"]
                     for line in receipt.read_text().splitlines() if line.strip()}
    pending = [path for path in paths if path.name not in completed]
    print(json.dumps({"images": len(paths), "already_uploaded": len(completed),
                      "pending": len(pending), "batch": args.batch,
                      "project": args.project, "tags": tags,
                      "execute": args.execute}, indent=2))
    if not args.execute:
        return

    workspace = get_workspace(args.workspace)
    project = open_project(workspace, args.project, args.create, args.annotation)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("a") as stream:
        for index, path in enumerate(pending, 1):
            metadata = {"annotation_status": "unlabelled"}
            if path.stem[-3:].isdigit():
                metadata["generation_manifest_id"] = int(path.stem[-3:])
            project.upload(
                image_path=str(path), split="train", num_retry_uploads=3,
                batch_name=args.batch, tag_names=tags, metadata=metadata)
            row = {"filename": path.name, "batch": args.batch,
                   "workspace": args.workspace, "project": args.project}
            stream.write(json.dumps(row) + "\n"); stream.flush()
            print(f"uploaded {index}/{len(pending)}: {path.name}")


if __name__ == "__main__":
    main()

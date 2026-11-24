#!/usr/bin/env python3
"""Create/update the private HF Space with the front end: python3 deploy/space/push.py [--repo abshetty/pattern-seg]."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi

D = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="abshetty/pattern-seg"); ap.add_argument("--token")
a = ap.parse_args()
api = HfApi(token=a.token)
api.create_repo(a.repo, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
files = {"Dockerfile": D / "space/Dockerfile", "README.md": D / "space/README.md", "requirements-app.txt": D / "requirements-app.txt",
         "deploy/app.py": D / "app.py", "deploy/app.html": D / "app.html"}
for dst, src in files.items():
    api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=a.repo, repo_type="space", commit_message=f"update {dst}")
print(f"https://huggingface.co/spaces/{a.repo}")

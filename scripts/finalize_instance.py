#!/usr/bin/env python3
"""End of the 2026-11-14 GH200 session (the user: back up, then terminate automatically). Run detached:

    nohup python3 scripts/finalize_instance.py > logs/finalize.log 2>&1 &

1. wait: the v2 self-training A/B has ended (logs/selftrain_ab_v2.log "SELFTRAIN AB DONE", or its script is gone) AND the
   Roboflow drafts are uploaded (logs/DRAFTS_UPLOADED) -- or DRAFTS_TIMEOUT_H after the A/B ended, whichever first;
2. git: v2 metrics, logs and the uploaded-draft records committed and pushed; HEAD must equal origin;
3. backup to the private Hub dataset BACKUP_REPO: blended checkpoints, each training arm's epoch_0.pth, the round-5
   discovery outputs (no symlinked duplicates), the un-augmented pseudo pools, the draft records; every local file's size
   is checked against the Hub listing;
4. terminate THIS instance (INSTANCE_ID) through the Lambda Cloud API, only if 2 and 3 passed and the API lists that id with
   this machine's public IP. Any failure stops before termination.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
INSTANCE_ID = "5764a1b50bda44b6a72254984e199176"      # gpu_1x_gh200 us-east-3, 192.222.51.135 (listed 2026-11-14)
BACKUP_REPO = "abshetty/floz-runs-backup"
DRAFTS_TIMEOUT_H = 3.0
API = "https://cloud.lambdalabs.com/api/v1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}: {r.stderr.strip()[-500:]}")
    return r.stdout.strip()


def env():
    for line in Path.home().joinpath(".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'"))


def wait():
    ab_log = ROOT / "logs/selftrain_ab_v2.log"
    ab_end = None
    while True:
        running = subprocess.run("pgrep -f run_selftrain_ab_v2.sh", shell=True, capture_output=True).returncode == 0
        done = ab_log.exists() and "SELFTRAIN AB DONE" in ab_log.read_text()
        if ab_end is None and ab_log.exists() and (done or not running):
            ab_end = time.time()
            log("v2 A/B ended", "DONE" if done else "WITHOUT the DONE line (failed?)")
        if ab_end is not None:
            if (ROOT / "logs/DRAFTS_UPLOADED").exists():
                log("drafts uploaded"); return
            if time.time() - ab_end > DRAFTS_TIMEOUT_H * 3600:
                log(f"drafts not uploaded {DRAFTS_TIMEOUT_H} h after the A/B: going on without them"); return
        time.sleep(60)


def git_sync():
    ab = (ROOT / "logs/selftrain_ab_v2.log").read_text() if (ROOT / "logs/selftrain_ab_v2.log").exists() else ""
    tail = [l for l in ab.splitlines() if l.startswith(("st vs", "st_s", "GATE"))]
    with open("synth_progress.md", "a") as f:
        f.write("\n\n## 2026-11-14 — self-training A/B v2 (auto-recorded by scripts/finalize_instance.py before termination)\n\n")
        f.write("\n".join(f"    {l}" for l in tail) if tail else "    (no gate lines: see logs/selftrain_ab_v2.log)")
        f.write(f"\n\nBackup of local-only checkpoints / round-5 discovery / pseudo pools: `{BACKUP_REPO}` (private). "
                f"Instance {INSTANCE_ID} terminated after the backup (the user's instruction).\n")
    sh("git add synth_progress.md")
    sh("git add -f logs/selftrain_ab_v2.log logs/selftrain_ab.log data/selftrain/drafts_uploaded "
       "$(ls -d data/evaluations/selftrain2_* data/evaluations/st2_r5_s* 2>/dev/null) 2>/dev/null || true", check=False)
    sh('GIT_AUTHOR_NAME=Claude GIT_AUTHOR_EMAIL=noreply@anthropic.com GIT_COMMITTER_NAME=Claude '
       'GIT_COMMITTER_EMAIL=noreply@anthropic.com git commit -qm "Self-training v2 results + logs, recorded before instance '
       'termination\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"', check=False)
    for _ in range(3):
        if subprocess.run("git pull -q --rebase && git push -q", shell=True).returncode == 0:
            break
        time.sleep(20)
    sh("git fetch -q")
    head, up = sh("git rev-parse HEAD"), sh("git rev-parse @{u}")
    if head != up:
        raise RuntimeError(f"git not in sync: HEAD {head} != upstream {up}")
    log("git in sync at", head[:7])


def backup():
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(BACKUP_REPO, repo_type="dataset", private=True, exist_ok=True)
    files = []
    runs = ROOT / "data/runs"
    files += [p for p in runs.glob("ck_*.pth")]
    files += [p / "epoch_0.pth" for p in runs.glob("ck_*_dr") if (p / "epoch_0.pth").exists()]
    files += [p / "epoch_0.pth" for p in runs.glob("ck_*_nodr") if (p / "epoch_0.pth").exists()]
    st = ROOT / "data/selftrain"
    for d in ("disc2_r5", "pool_r5", "pool_r5_v2", "drafts_uploaded"):
        files += [p for p in (st / d).rglob("*") if p.is_file()]
    allr = st / "disc2_r5_all"
    if allr.exists():
        for sub in allr.iterdir():
            if not sub.is_symlink():
                files += [p for p in sub.rglob("*") if p.is_file()]
    files += [st / x for x in ("r5_elevations.txt", "r5_other.txt") if (st / x).exists()]
    rel = sorted({str(p.relative_to(ROOT / "data")) for p in files})
    stage = ROOT / "data/_backup_stage"          # hard links to just these files (data/ itself holds ~10^6 pool files)
    for r in rel:
        dst = stage / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(os.path.realpath(ROOT / "data" / r), dst)
    log(f"backup: {len(rel)} files, {sum((stage / r).stat().st_size for r in rel) / 2**30:.1f} GB -> {BACKUP_REPO}")
    for top in sorted({r.split("/")[0] for r in rel}):          # one commit per top-level folder, retried
        for attempt in range(3):
            try:
                api.upload_folder(repo_id=BACKUP_REPO, repo_type="dataset", folder_path=str(stage / top), path_in_repo=top,
                                  commit_message=f"backup {top} (2026-11-14, before instance termination)")
                break
            except Exception as e:
                log(f"upload {top} attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(60)
    remote = {f.path: f.size for f in api.list_repo_tree(BACKUP_REPO, repo_type="dataset", recursive=True)
              if type(f).__name__ == "RepoFile"}
    bad = [r for r in rel if remote.get(r) != (stage / r).stat().st_size]
    if bad:
        raise RuntimeError(f"backup verification failed for {len(bad)} files, e.g. {bad[:3]}")
    log(f"backup verified: {len(rel)} files match by size")


def terminate():
    import requests
    key = os.environ["LAMBDA_API_KEY"]
    ip = requests.get("https://api.ipify.org", timeout=20).text.strip()
    inst = requests.get(f"{API}/instances", auth=(key, ""), timeout=30).json()["data"]
    me = [x for x in inst if x["id"] == INSTANCE_ID]
    if not me or me[0].get("ip") != ip:
        raise RuntimeError(f"refusing to terminate: {INSTANCE_ID} not listed with this machine's IP {ip}")
    log(f"terminating {INSTANCE_ID} ({me[0]['instance_type']['name']}, {ip})")
    r = requests.post(f"{API}/instance-operations/terminate", auth=(key, ""), json={"instance_ids": [INSTANCE_ID]},
                      timeout=60)
    log("terminate response:", r.status_code, r.text[:300])


if __name__ == "__main__":
    env()
    if "--backup-only" in sys.argv:          # start the upload early (network-bound); the final run re-checks / tops it up
        backup()
        sys.exit(0)
    try:
        wait()
        git_sync()
        backup()
    except Exception as e:
        log("STOPPED before termination:", e)
        sys.exit(1)
    terminate()

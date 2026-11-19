#!/usr/bin/env python3
"""End of the 2026-11-19 GH200 session (the user: document, push, back up, then terminate this VM automatically). Detached:

    nohup python3 scripts/finalize_session.py > logs/finalize.log 2>&1 &

1. wait   gpu_queue.sh logged "QUEUE DONE", then the machine is QUIET for QUIET_MIN consecutive minutes: no GPU compute
          process and no training / scoring / Gemini job (train_refunet, evaluate_refunet_selection, discover, gemini_refine,
          pseudolabel, augment_local_dataset, benchmark_frozen_candidate);
2. docs   logs/DOCS_DONE (written by the Claude session once the results are documented in synth_progress.md / codex_doc.md /
          startup.md and pushed); if it has not appeared DOCS_TIMEOUT_H after quiet, the raw result lines are appended to
          synth_progress.md here so nothing is lost;
3. git    commit (docs, scripts, small result files), pull --rebase, push; then CHECK: HEAD == upstream and no tracked file
          modified. This check is the gate for everything after it;
4. backup this session's local-only checkpoints / discovery / verdicts / pools to the private Hub dataset BACKUP_REPO, every
          file's size verified against the Hub listing;
5. kill   terminate THIS instance through the Lambda Cloud API, only if the API lists INSTANCE_ID with this machine's public
          IP; then poll the API until it reports the instance terminating / terminated (logged).
Any failure before 5 stops the script with "STOPPED before termination" (the heartbeat reports it); the VM is left running.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
INSTANCE_ID = "93989cf7a6714261aac29a1903ace194"      # gpu_1x_gh200 us-east-3, 192.222.59.224 (listed 2026-11-19)
BACKUP_REPO = "abshetty/floz-runs-backup"
API = "https://cloud.lambdalabs.com/api/v1"
QUIET_MIN = 10
DOCS_TIMEOUT_H = 2.0
BUSY = ("train_refunet", "evaluate_refunet_selection", "selftrain/discover", "gemini_refine", "pseudolabel.py",
        "augment_local_dataset", "benchmark_frozen_candidate", "run_ab_vs_best", "run_selftrain_ab", "gpu_queue.sh")
STATE = ROOT / "logs/finalize_state.json"


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def state(**kw):
    s = json.loads(STATE.read_text()) if STATE.exists() else {}
    s.update(kw, updated=time.strftime("%F %T"))
    STATE.write_text(json.dumps(s, indent=1))


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


def busy():
    """Names of running jobs (read from /proc, so this process and shells that merely mention a name do not count)."""
    found = set()
    gpu = sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader", check=False).split()
    if gpu:
        found.add(f"gpu pids {','.join(gpu)}")
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        exe = argv[0].decode(errors="ignore").rsplit("/", 1)[-1]
        if exe not in ("python3", "python", "bash") or (len(argv) > 1 and argv[1] == b"-c"):
            continue                                  # "bash -c '<command text>'" only mentions names; it runs nothing
        script = " ".join(a.decode(errors="ignore") for a in argv[1:3])
        for b in BUSY:
            if b in script:
                found.add(b)
    return found


def wait():
    q = ROOT / "logs/gpu_queue.log"
    while not (q.exists() and "QUEUE DONE" in q.read_text()):
        state(step="wait: queue"); time.sleep(60)
    log("queue done")
    quiet_since = None
    while True:
        b = busy()
        if b:
            quiet_since = None; state(step="wait: quiet", busy=sorted(b))
        else:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= QUIET_MIN * 60:
                break
            state(step=f"wait: quiet {int(time.time() - quiet_since)} s", busy=[])
        time.sleep(30)
    log(f"quiet for {QUIET_MIN} min")
    t0 = time.time()
    while not (ROOT / "logs/DOCS_DONE").exists():
        if time.time() - t0 > DOCS_TIMEOUT_H * 3600:
            log("DOCS_DONE missing after the timeout: recording raw results"); auto_record(); return
        state(step="wait: DOCS_DONE"); time.sleep(60)
    log("docs done (logs/DOCS_DONE)")


def auto_record():
    lines = []
    for f in ("logs/selftrain_ab_v3.log", "logs/fg3seed_compare.txt", "logs/refunf_compare.txt", "logs/gpu_queue.log"):
        p = ROOT / f
        if p.exists():
            keep = [l for l in p.read_text().splitlines() if any(k in l for k in ("GATE", "vs", "paired mean", "DONE", "FAILED"))]
            lines += [f"`{f}`:"] + [f"    {l}" for l in keep[-40:]]
    fb = Path.home() / "frozen_ab/benchmark.log"
    if fb.exists():
        lines += ["frozen benchmark (`~/frozen_ab/benchmark.log`, tail):"] + [f"    {l}" for l in fb.read_text().splitlines()[-40:]]
    with open("synth_progress.md", "a") as f:
        f.write("\n\n## 2026-11-19 -- GPU queue results (auto-recorded by scripts/finalize_session.py: the session did not "
                "document them in time)\n\n" + "\n".join(lines) + "\n")


def git_sync():
    sh("git add synth_progress.md codex_doc.md startup.md PROJECT_UNDERSTANDING.md 2>/dev/null || true", check=False)
    sh("git add scripts run_*.sh gpu_queue.sh eval_labels eval_boxes 2>/dev/null || true", check=False)
    small = ["logs/selftrain_ab_v3.log", "logs/fg3seed_compare.txt", "logs/refunf_compare.txt", "logs/gpu_queue.log",
             "logs/fg_seed9.log", "logs/gref_validate.log", "logs/gref_validate_hydrogen.log", "logs/gref_validate_flash.log",
             "logs/finalize.log"]
    small += [str(p) for pat in ("fgab_*_s9", "fg3_*", "selftrainv3_*", "selftrain_ctl*", "stv3_r5_s*", "refunf*", "fgavg78_head")
              for p in (ROOT / "data/evaluations").glob(pat)]
    for d in sorted((Path.home() / "frozen_ab").glob("frozen-benchmark-*")):      # the codex handoff's return directory
        sh(f"mkdir -p data/evaluations/frozen_benchmark && cp -r '{d}' data/evaluations/frozen_benchmark/", check=False)
    small += [str(p) for p in (ROOT / "data/evaluations").glob("frozen_benchmark")]
    small += [str(p) for p in (ROOT / "data/selftrain").glob("rep_*.json")]
    small += [str(p) for p in (ROOT / "data/selftrain").glob("gref_*")]
    small += [str(p) for p in (ROOT / "data/selftrain").glob("pool_r5_v3/selftrain_manifest.json")]
    for s in small:                                   # result files only: never the *.json.maps probability dumps
        p = Path(s)
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and ".maps" not in str(f)
                                         and f.suffix in (".json", ".jsonl", ".log", ".txt")] if p.exists() else []
        for f in files:
            sh(f"git add -f '{f}'", check=False)
    msg = ("Session 2026-11-19 results, docs and logs, recorded before instance termination\n\n"
           "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    sh(f"git -c user.name=Claude -c user.email=noreply@anthropic.com commit -qm {json.dumps(msg)}", check=False)
    for attempt in range(5):
        if subprocess.run("git pull -q --rebase && git push -q", shell=True).returncode == 0:
            break
        log(f"push attempt {attempt + 1} failed"); time.sleep(30)
    check_git()


def check_git():
    sh("git fetch -q")
    head, up = sh("git rev-parse HEAD"), sh("git rev-parse @{u}")
    dirty = sh("git status --porcelain --untracked-files=no")
    if head != up or dirty:
        raise RuntimeError(f"git NOT in sync: HEAD {head[:7]} upstream {up[:7]} dirty={bool(dirty)} {dirty[:200]}")
    log("git verified: HEAD == upstream", head[:7], "and no tracked changes"); state(git_verified=head)


def backup():
    from huggingface_hub import HfApi
    api = HfApi()
    files = []
    runs = ROOT / "data/runs"
    for pat in ("ck_*s9*.pth", "ck_*3seed*.pth", "ck_stv3*.pth", "ck_refunf*.pth"):
        files += list(runs.glob(pat))
    for pat in ("ck_*s9_dr", "ck_*s9_nodr", "ck_stv3*_dr", "ck_stv3*_nodr", "ck_refunf*_dr", "ck_refunf*_nodr"):
        files += [p / "epoch_0.pth" for p in runs.glob(pat) if (p / "epoch_0.pth").exists()]
    st = ROOT / "data/selftrain"
    for d in ("disc_gemini30", "disc_val", "pool_r5_v3"):
        files += [p for p in (st / d).rglob("*") if p.is_file()] if (st / d).exists() else []
    for d in st.glob("gref_*"):
        files += [p for p in d.rglob("*") if p.is_file()]
    files += list(st.glob("rep_*.json"))
    rel = sorted({str(p.relative_to(ROOT / "data")) for p in files})
    stage = ROOT / "data/_backup_stage_1119"
    for r in rel:
        dst = stage / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(os.path.realpath(ROOT / "data" / r), dst)
    log(f"backup: {len(rel)} files, {sum((stage / r).stat().st_size for r in rel) / 2**30:.1f} GB -> {BACKUP_REPO}")
    for top in sorted({r.split("/")[0] for r in rel}):
        for attempt in range(3):
            try:
                api.upload_folder(repo_id=BACKUP_REPO, repo_type="dataset", folder_path=str(stage / top), path_in_repo=top,
                                  commit_message=f"backup {top} (2026-11-19, before instance termination)")
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
    log(f"backup verified: {len(rel)} files match by size"); state(backup_verified=len(rel))


def terminate():
    import requests
    key = os.environ["LAMBDA_API_KEY"]
    ip = requests.get("https://api.ipify.org", timeout=20).text.strip()
    inst = requests.get(f"{API}/instances", auth=(key, ""), timeout=30).json()["data"]
    me = [x for x in inst if x["id"] == INSTANCE_ID]
    if not me or me[0].get("ip") != ip:
        raise RuntimeError(f"refusing to terminate: {INSTANCE_ID} not listed with this machine's IP {ip}")
    check_git()                                      # last look: nothing committed since, push still verified
    log(f"terminating {INSTANCE_ID} ({me[0]['instance_type']['name']}, {ip})"); state(step="terminating")
    r = requests.post(f"{API}/instance-operations/terminate", auth=(key, ""), json={"instance_ids": [INSTANCE_ID]},
                      timeout=60)
    log("terminate response:", r.status_code, r.text[:300])
    for _ in range(20):                               # the API's own confirmation, while this machine still runs
        time.sleep(15)
        try:
            s = [x.get("status") for x in requests.get(f"{API}/instances", auth=(key, ""), timeout=30).json()["data"]
                 if x["id"] == INSTANCE_ID]
        except Exception as e:
            s = [f"error {e}"]
        log("instance status:", s[0] if s else "no longer listed"); state(step=f"terminate: status {s[0] if s else 'gone'}")
        if not s or s[0] in ("terminating", "terminated"):
            break


if __name__ == "__main__":
    env()
    if "--check-git" in sys.argv:
        check_git(); sys.exit(0)
    try:
        wait()
        git_sync()
        backup()
    except Exception as e:
        log("STOPPED before termination:", e); state(step="STOPPED", error=str(e)[:500])
        sys.exit(1)
    terminate()

#!/usr/bin/env python3
"""Multi-seed divergence probe — the hardened eval instrument.

WHY THIS EXISTS: a single 10-epoch / 28-real / single-seed run of train.py has a
real_iou noise band of ~0.01-0.07 driven by INTRINSIC training stochasticity
(nondeterministic CUDA kernels + bf16 autocast) that seeding does NOT remove
(verified: two same-seed runs diverge). At real_iou ~= 0.2 that noise is larger
than the generator effects we want to measure, so single-run div deltas < ~0.05
are NOISE. This harness runs train.py over N seeds and reports mean +/- std so an
effect can be trusted only when it clears the error bar.

It also collapses within-run epoch jitter by averaging each seed's last P epochs
("plateau mean") before taking the across-seed mean/std.

Usage:
  python scripts/probe_multiseed.py --local-data /tmp/synth_X_500 \
      --seeds 0,1,2,3,4 --epochs 10 --tag baseline --extra "--domain-random"

Compare two generator configs by running it twice (different --local-data/--tag)
and checking whether the div mean+/-std intervals overlap.
"""
import argparse
import re
import statistics
import subprocess
from pathlib import Path

EPOCH_RE = re.compile(
    r"Epoch (\d+):.*?synth_iou ([\d.]+).*?real_iou ([\d.]+).*?DIVERGENCE ([+-][\d.]+)"
)


def parse_log(path):
    """Return {epoch: (synth, real, div)} from a train.py log."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = EPOCH_RE.search(line)
        if m:
            ep = int(m.group(1))
            out[ep] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def ms(xs):
    """mean, std (population) of a list."""
    return (sum(xs) / len(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-data", required=True)
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--tag", required=True, help="label for log files / output")
    ap.add_argument("--image-max-size", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--plateau", type=int, default=3,
                    help="average each seed's last P epochs before across-seed stats")
    ap.add_argument("--extra", default="--domain-random",
                    help="extra args passed verbatim to train.py")
    ap.add_argument("--log-root", default="/tmp")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse a seed's log if it already has all epochs")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    logs = {}
    for s in seeds:
        log = f"{args.log_root}/probe_{args.tag}_s{s}.log"
        have = parse_log(log) if Path(log).exists() else {}
        if args.skip_existing and len(have) >= args.epochs:
            print(f"[seed {s}] reuse {log} ({len(have)} epochs)")
            logs[s] = have
            continue
        cmd = [
            "python", "train.py", "--local-data", args.local_data,
            "--image-max-size", str(args.image_max_size),
            "--batch-size", str(args.batch_size), "--epochs", str(args.epochs),
            "--num-workers", str(args.num_workers), "--real-eval",
            "--seed", str(s),
            "--checkpoint-dir", f"{args.log_root}/ck_probe_{args.tag}_s{s}",
            "--log-dir", f"{args.log_root}/tb_probe_{args.tag}_s{s}",
        ] + args.extra.split()
        env = {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "MPLCONFIGDIR": "/tmp/matplotlib"}
        import os
        full_env = {**os.environ, **env}
        print(f"[seed {s}] running -> {log}")
        with open(log, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=full_env,
                           check=True)
        logs[s] = parse_log(log)

    # Per-epoch across-seed stats
    print(f"\n==== probe '{args.tag}'  ({len(seeds)} seeds: {seeds}) ====")
    print(f"{'epoch':>5} | {'synth mean±std':>18} | {'real mean±std':>18} | "
          f"{'div mean±std':>18}")
    all_eps = sorted(set().union(*[set(l) for l in logs.values()]))
    for ep in all_eps:
        s_vals = [logs[s][ep][0] for s in seeds if ep in logs[s]]
        r_vals = [logs[s][ep][1] for s in seeds if ep in logs[s]]
        d_vals = [logs[s][ep][2] for s in seeds if ep in logs[s]]
        sm, ss = ms(s_vals); rm, rs = ms(r_vals); dm, ds = ms(d_vals)
        print(f"{ep:>5} | {sm:>8.4f}±{ss:<8.4f} | {rm:>8.4f}±{rs:<8.4f} | "
              f"{dm:>+8.4f}±{ds:<8.4f}")

    # Plateau summary: mean of last P epochs per seed, then across-seed mean±std.
    print(f"\n---- plateau (mean of last {args.plateau} epochs per seed) ----")
    plat_real, plat_div, plat_synth = [], [], []
    for s in seeds:
        eps = sorted(logs[s])[-args.plateau:]
        if not eps:
            continue
        plat_synth.append(sum(logs[s][e][0] for e in eps) / len(eps))
        plat_real.append(sum(logs[s][e][1] for e in eps) / len(eps))
        plat_div.append(sum(logs[s][e][2] for e in eps) / len(eps))
    sm, ss = ms(plat_synth); rm, rs = ms(plat_real); dm, ds = ms(plat_div)
    print(f"  synth_iou = {sm:.4f} ± {ss:.4f}")
    print(f"  real_iou  = {rm:.4f} ± {rs:.4f}")
    print(f"  DIVERGENCE = {dm:+.4f} ± {ds:.4f}   (n={len(plat_div)} seeds)")
    print(f"\n  -> trust a config difference only if its plateau div interval "
          f"clears this ±{ds:.4f} band.")


if __name__ == "__main__":
    main()

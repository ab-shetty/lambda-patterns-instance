# Mined pool manifests

`build_hard_pool.py` output is not reproducible from the generator alone: which fresh sheets were
mined depends on the miner model's scores. These manifests (`mined_files`) pin the exact sheets.

Rebuild a pool without re-mining:
1. Generate its base and fresh pools with the committed generator (`run_hard_mine.sh` shows the
   flags; since 2026-10-04 ALSO pass `--plan-v2 1 --cut-standing 1 --downspouts 0 --ref-exclude 0`,
   the defaults these pools were built with; pools from the 2026-10-04 defaults (`*_v1004`,
   `revitfailswiss_*`) need `--near-pairs 0 --near-pairs-plan 0 --balcony-doors 0` since 2026-10-07,
   and the older ones need both sets): `revitcur_*` = the 2026-10-02 defaults, `revitfail_*` = + `--roof-lines 0.3 --railings 0.15
   --soft-shadows 0.5 --faint-lines 0.2`; base ids 0-1999 n=2000, fresh ids 100000+ with
   n=6000 (cur) / 4000 (fail), seed 6).
2. Hardlink the base pool's first 1,000 sheets (sorted) plus every file in `mined_files` from the
   fresh pool into one images/ + annotations/ directory.

`revitfail_hard2000` is the pool behind the synthetic-only 0.8668 model and the 0.8816 fine-tune.

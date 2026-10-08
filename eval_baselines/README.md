# Baseline per-question results (2026-10-06, GH200)

Per-question IoUs so a new model can be compared paired (`scripts/paired_compare.py --a <baseline> --b <new>`)
without re-scoring the baseline (~8 min per model at 4096 on CPU). Hand boxes (HF14 / val: eval_boxes/hand_v1.json;
Gemini30: eval_boxes/gemini30_v1.json on data/eval_pools/gemini30_s20261006 built with r4 v2, see
pool_manifests/gemini30_s20261006.json). Threshold 0.35. Compare at the same inference size.

| file | model (Hub, private) | mean |
|---|---|---:|
| longswiss-swa37-41_{hf14,val,gemini30}_4096 | abshetty/floz-refunet-swint-longswiss-swa37-41 (best synthetic) | 0.8779 / 0.8766 / 0.8295 |
| revitfailswiss2k-e8_{hf14,val,gemini30}_2048 | abshetty/floz-refunet-swint-revitfailswiss2k-e8 (2k screen control) | 0.8090 / 0.7559 / 0.7239 |

2026-10-07: Gemini30 sheet 16 (r3 #048) was relabelled by the user (Roboflow r3 v3: one missed pattern3 piece added,
appended as question 23 so every other automatic box is unchanged). The longswiss Gemini30 file has that sheet's rows
rescored on CPU at 4096 (0.8303 -> 0.8622 on the sheet; Gemini30 0.8255 -> 0.8295, 195 questions). The
revitfailswiss2k-e8 Gemini30 file got the same treatment at 2048 on 2026-10-08 (sheet 16 0.3656 -> 0.3653; Gemini30
0.7258 -> 0.7239).

Invariance probe (`scripts/probes/`, leak into the other panel per attribute): `probe_longswiss-swa37-41_4096.json`,
`probe_revitfailswiss2k-e8_2048.json`. Print a new model's table with `scripts/probes/invariance_leak.py`.

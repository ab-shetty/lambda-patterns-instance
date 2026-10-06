# Baseline per-question results (2026-10-06, GH200)

Per-question IoUs so a new model can be compared paired (`scripts/paired_compare.py --a <baseline> --b <new>`)
without re-scoring the baseline (~8 min per model at 4096 on CPU). Hand boxes (HF14 / val: eval_boxes/hand_v1.json;
Gemini30: eval_boxes/gemini30_v1.json on data/eval_pools/gemini30_s20261006 built with r4 v2, see
pool_manifests/gemini30_s20261006.json). Threshold 0.35. Compare at the same inference size.

| file | model (Hub, private) | mean |
|---|---|---:|
| longswiss-swa37-41_{hf14,val,gemini30}_4096 | abshetty/floz-refunet-swint-longswiss-swa37-41 (best synthetic) | 0.8779 / 0.8766 / 0.8255 |
| revitfailswiss2k-e8_{hf14,val,gemini30}_2048 | abshetty/floz-refunet-swint-revitfailswiss2k-e8 (2k screen control) | 0.8090 / 0.7559 / 0.7258 |

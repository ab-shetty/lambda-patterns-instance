# Baseline per-question results (2026-10-06, GH200)

Per-question IoUs so a new model can be compared paired (`scripts/paired_compare.py --a <baseline> --b <new>`)
without re-scoring the baseline (~8 min per model at 4096 on CPU). Hand boxes (HF14 / val: eval_boxes/hand_v1.json;
Gemini30: eval_boxes/gemini30_v1.json on data/eval_pools/gemini30_s20261006 built with r4 v2, see
pool_manifests/gemini30_s20261006.json). Threshold 0.35. Compare at the same inference size.

| file | model (Hub, private) | mean |
|---|---|---:|
| longswiss-swa37-41_{hf14,val,gemini30}_4096 | abshetty/floz-refunet-swint-longswiss-swa37-41 (best synthetic) | 0.8779 / 0.8766 / 0.8295 |
| revitfailswiss2k-e8_{hf14,val,gemini30}_2048 | abshetty/floz-refunet-swint-revitfailswiss2k-e8 (2k screen control) | 0.8090 / 0.7559 / 0.7239 |
| revithue2k-e8_{hf14,val,gemini30}_2048 | not published (2026-10-13 hue-pair screen; checkpoint `data/runs/ck_2k_revithueroi_r2048_s7/epoch_8.pth` on the GH200) | 0.7806 / 0.7161 / 0.7068 |
| revitwing2k-e8_{hf14,val,gemini30}_2048 | not published (2026-10-14 wing colour-pair screen, passed the gate; `data/runs/ck_2k_revitwingroi_r2048_s7/epoch_8.pth` on the GH200) | 0.7920 / 0.7527 / 0.7647 |
| revitgable2k-e8_{hf14,val,gemini30}_2048 | not published (2026-10-14 gable-trim screen; `data/runs/ck_2k_revitgableroi_r2048_s7/epoch_8.pth` on the GH200) | 0.8264 / 0.7786 / 0.7753 |
| single18kq-blend30_{hf14,hf14fix2,val,gemini30}_2048 | abshetty/floz-refunet-swint-single18kq-blend30 (best) at 2048, for the scale gap (2026-10-17) | 0.8582 / 0.8764 / 0.8566 / 0.8750 |
| single18kms-blend{30,50}_{hf14,hf14fix2,val,gemini30}_4096, -blend30_*_2048 | arms `abshetty/floz-refunet-swint-single18kms-{dr,nodr}` (2026-10-17 multi-res, `--image-max-size-max 3072`, failed the gate alone) | b30 @4096 0.9089 / 0.9263 / 0.8980 / 0.8652 |
| single18kw-blend{30,50}_{hf14,hf14fix2,val,gemini30}_4096 | arms `abshetty/floz-refunet-swint-single18kw-{dr,nodr}` (2026-10-17 wing colour pairs, failed the gate) | b30 0.9146 / 0.9224 / 0.8997 / 0.8495 |
| **xblend-kq30-kms30**_{hf14,hf14fix2,val,gemini30}_4096 | **abshetty/floz-refunet-swint-xblend-kq30-kms30 (best, 2026-10-17)** | 0.9163 / 0.9290 / 0.8988 / 0.8740 |
| xblend-kq30-kw30_{hf14,hf14fix2,val,gemini30}_4096 | not published (0.5 kq30 + 0.5 kw30, failed the gate) | 0.9156 / 0.9225 / 0.8994 / 0.8680 |
| single18kg-blend{30,50}_{hf14,hf14fix2,val,gemini30}_4096 | arms `abshetty/floz-refunet-swint-single18kg-{dr,nodr}` (2026-10-17 gable tone pairs + twins, GEN 2026-10-17; failed the gate) | b30 0.9223 / 0.9262 / 0.8878 / 0.8613 |
| xblend-xa-kg30_{hf14,hf14fix2,val,gemini30}_4096 | not published (0.5 xblend-kq30-kms30 + 0.5 kg30, failed the gate) | 0.9231 / 0.9296 / 0.8985 / 0.8691 |
| single36k-blend{30,50}_{hf14,hf14fix2,val,gemini30}_4096 | arms `abshetty/floz-refunet-swint-single36k-{dr,nodr}` (2026-10-17, 35,812 sheets at the kq config, one epoch; failed the gate) | b30 0.9048 / 0.9211 / 0.8912 / 0.8704 |
| xblend-xa-s36k30_{hf14,hf14fix2,val,gemini30}_4096 | not published (0.5 xblend-kq30-kms30 + 0.5 s36k30; passes the gate's letter at noise level, held for the user) | 0.9126 / 0.9293 / 0.8976 / 0.8750 |

2026-10-07: Gemini30 sheet 16 (r3 #048) was relabelled by the user (Roboflow r3 v3: one missed pattern3 piece added,
appended as question 23 so every other automatic box is unchanged). The longswiss Gemini30 file has that sheet's rows
rescored on CPU at 4096 (0.8303 -> 0.8622 on the sheet; Gemini30 0.8255 -> 0.8295, 195 questions). The
revitfailswiss2k-e8 Gemini30 file got the same treatment at 2048 on 2026-10-08 (sheet 16 0.3656 -> 0.3653; Gemini30
0.7258 -> 0.7239).

Invariance probe (`scripts/probes/`, leak into the other panel per attribute): `probe_longswiss-swa37-41_4096.json`,
`probe_revitfailswiss2k-e8_2048.json`, `probe_revithue2k-e8_2048.json`, `probe_revitwing2k-e8_2048.json`. Print a new model's table with `scripts/probes/invariance_leak.py`.

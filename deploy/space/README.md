---
title: Pattern Segmentation
emoji: 🧱
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

Box-and-see front end for the pattern-segmentation product (source: `deploy/` in the GitHub repo). Pick one of the 28 real
sheets, drag a box, get the polygons; the model runs on a Runpod Serverless GPU endpoint. Needs the Space secrets
`HF_TOKEN` and `RUNPOD_API_KEY`.

# Vehicle Re-Identification under Image Degradation

How well does a vehicle re-identification model hold up on low-quality, real-world traffic footage, and does training for it help?

Models are [torchreid](https://github.com/KaiyangZhou/deep-person-reid) OSNet x1_0 (ImageNet-pretrained), trained on **VRIC** (small, low-quality crops from traffic video) and **VeRi-776** (higher-resolution surveillance images).

## Key findings

1. **The original evaluation overstated accuracy by about 1.8×.** Testing against a 100-vehicle gallery gave 68.3 mAP on VRIC. Against the full 2,811-image gallery, the same model scores **37.5 mAP [95% CI 35.9, 39.0]**. All results below use the full gallery.
2. **"Blur" was mostly a resolution effect.** The standard blur score (Laplacian variance) depends on image size. It correlates with resolution at Spearman ρ = −0.49 (VRIC) and −0.57 (VeRi), so the "blurry" group was mostly large images. That's why blurry images *seemed* to retrieve better (41.7 vs. 33.3 mAP). With size normalized and resolution held fixed, sharper images do retrieve better, but the effect is small (partial ρ = 0.05–0.10).
3. **Low-resolution augmentation helps on low-quality data, but not on high-quality data.** Random downscale-then-upscale during training (parameters fixed before any test results):
   - **VRIC: +2.6 mAP [1.9, 3.3]**, with every query group improving.
   - **VeRi-776: −0.3 mAP [−0.7, −0.0]**, a slight drop. Its images are about 280 px, so the augmentation adds a mismatch instead of fixing one.
   - On VRIC the gain is broad rather than concentrated on low-res queries: the gap shrinks by only 1.2 [−0.4, 2.6].

## Results

300 training identities, 5 epochs, 3 seeds per setting. mAP is averaged over seeds, with 95% bootstrap CIs over queries (1,000 resamples). Δ is paired: the same queries, augmented minus baseline.

| Dataset | Query group | n | Baseline mAP | Low-res aug mAP | Δ mAP |
|---|---|---:|---|---|---|
| VRIC | all queries | 2811 | 36.7 [35.3, 38.1] | 39.3 [37.9, 40.6] | **+2.6 [1.9, 3.3]** |
| VRIC | low-res half | 1404 | 31.6 [29.8, 33.4] | 34.8 [33.0, 36.6] | +3.2 [2.1, 4.2] |
| VRIC | high-res half | 1407 | 41.8 [39.9, 44.1] | 43.8 [41.8, 45.9] | +2.0 [1.0, 3.0] |
| VeRi-776 | all queries | 1678 | 45.7 [44.6, 46.7] | 45.3 [44.2, 46.4] | −0.3 [−0.7, −0.0] |
| VeRi-776 | low-res half | 839 | 43.7 [42.2, 45.1] | 42.9 [41.3, 44.4] | −0.8 [−1.3, −0.3] |
| VeRi-776 | high-res half | 839 | 47.6 [46.1, 49.0] | 47.7 [46.2, 49.1] | +0.1 [−0.4, 0.6] |

Seed-to-seed standard deviation is ≤ 0.7 mAP for full-query results. The notebook also has per-group tables (a 2×2 split of resolution × size-normalized blur) and a secondary augmentation variant with Gaussian blur, which was no better.

## Production / deployment analysis

The research is framed as **model robustness under distribution shift**: how retrieval degrades when real-world footage is lower quality than training data. On top of the experiments above, the notebook includes:

- **Quality gate with human-in-the-loop deferral.** Each query gets a quality score from its own image (resolution plus size-normalized sharpness; no labels). The lowest-quality 10/20/30% are routed to human review instead of trusting the model, and the notebook reports mAP on the queries the model keeps, with bootstrap CIs. It compares against random deferral (no gain) and an oracle that uses labels (upper bound). The gate score was fixed before looking at results. Output: `quality_gate.csv`, `quality_gate.png`.
- **Inference cost.** Measured latency (ms per image at batch 1/32/256, fp32 and fp16) and throughput on a T4. Also: time to embed the full 11.6k-image VeRi gallery and time to rank one query against it. Output: `inference_speed.json`.
- **Experiment tracking with MLflow.** All 18 training runs are logged with their config (dataset, augmentation, seed, hyperparameters), full-gallery mAP / R1 / R5 for every query group, and training curves. A summary run adds the deployment numbers. Output: `mlflow.db`; view it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Method notes and caveats

- **Evaluation:** Market-1501 protocol. Euclidean distance; gallery images with the same ID *and* camera as the query are excluded.
- **Leakage checks:** train/test identity overlap is 0 on both datasets. VeRi query images also appear in its test folder, but always from the same camera, so the protocol removes them.
- **VRIC specifics:** every query and its match come from the same video. There is one gallery image per identity, so mAP equals mean reciprocal rank.
- **VeRi CIs are somewhat narrow:** its 1,678 queries come from about 200 vehicles and aren't independent.
- **Small-data setting:** 300 identities and 5 epochs. Absolute numbers are well below published full-dataset results; the findings are about relative effects.
- **Thresholds:** quality thresholds are median splits of query-image statistics, with no labels used. The one pre-specified test is baseline vs. low-res augmentation on all queries.

## Layout

```
notebooks/ReIdTraining.ipynb   # all experiments (subsets, training, evaluation, analysis); "Run All" on Colab
data/                          # local dataset copies (git-ignored, not in this repo)
```

## Running

The notebook is meant for **Google Colab with a T4 GPU** (works through the Colab VS Code extension).

1. Put the datasets on Google Drive:
   ```
   MyDrive/Colab Schoolwork/CAHSI 2026/Data/
   ├── VRIC/   (train_images/, probe_images/, gallery_images/, vric_*.txt)
   └── VeRi/   (image_train/, image_query/, image_test/, ...)
   ```
2. If your folder is somewhere else, change `DRIVE_BASE` in the **CONFIG** cell.
3. **Run All.** The first full run takes about 2 hours: 18 training runs, then evaluation.
   - A checkpoint is saved to Drive after every epoch, so after a disconnect, Run All again and it picks up from the last saved epoch.
   - Finished runs and evaluations are cached, so later runs only take minutes.

## Datasets

The datasets are not redistributed here. Request them from their authors:
- **VRIC**: Kanacı et al., *Vehicle Re-Identification in Context* (GCPR 2018)
- **VeRi-776**: Liu et al., *A Deep Learning-Based Approach to Progressive Vehicle Re-identification for Urban Surveillance* (ECCV 2016)

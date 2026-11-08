# Vehicle Re-Identification under Image Degradation

How robust is vehicle re-identification to low-quality, real-world traffic footage, and does training for it help? Models are [torchreid](https://github.com/KaiyangZhou/deep-person-reid) OSNet x1_0 (ImageNet-pretrained), trained on **VRIC** (small, low-quality traffic crops) and **VeRi-776** (higher-resolution surveillance images).

## Key findings

- **Full-gallery evaluation.** A 100-vehicle test subset gave 68.3 mAP on VRIC. The full 2,811-image gallery gives **37.5 mAP [95% CI 35.9, 39.0]**, which is the realistic number. All results use the full gallery.
- **Blur vs. resolution.** The standard blur score depends on image size (Spearman ρ = −0.49 with resolution on VRIC), so "blurry" images were mostly large ones. With resolution held fixed, sharper images retrieve slightly better (partial ρ = 0.05–0.10).
- **Low-resolution augmentation helps on low-quality data:** **+2.6 mAP [1.9, 3.3]** on VRIC. It doesn't transfer to high-resolution VeRi-776 (−0.3 [−0.7, −0.0]).

## Results

300 training identities, 5 epochs, 3 seeds per setting, 95% bootstrap CIs over queries.

| Dataset | Queries | Baseline mAP | Low-res aug mAP | Δ mAP |
|---|---|---|---|---|
| VRIC | all (2811) | 36.7 [35.3, 38.1] | 39.3 [37.9, 40.6] | **+2.6 [1.9, 3.3]** |
| VRIC | low-res half | 31.6 [29.8, 33.4] | 34.8 [33.0, 36.6] | +3.2 [2.1, 4.2] |
| VRIC | high-res half | 41.8 [39.9, 44.1] | 43.8 [41.8, 45.9] | +2.0 [1.0, 3.0] |
| VeRi-776 | all (1678) | 45.7 [44.6, 46.7] | 45.3 [44.2, 46.4] | −0.3 [−0.7, −0.0] |

## Deployment analysis

- **Quality gate:** routes the lowest-quality 10–30% of queries to human review, using a label-free score from resolution and sharpness. It reports mAP on the queries the model keeps, compared against random and oracle deferral.
- **Inference cost:** latency and throughput on a T4 (fp32/fp16, batch 1–256), plus gallery embedding and ranking time.
- **Experiment tracking:** all 18 training runs logged in MLflow (config, per-group metrics, training curves).

## Caveats

- VRIC queries and matches come from the same video, with one gallery image per vehicle.
- VeRi queries aren't independent (about 200 vehicles), so its CIs are slightly narrow.
- Small-data setting (300 identities, 5 epochs): the findings are about relative effects.
- No train/test identity overlap. Augmentation settings and quality thresholds were fixed before seeing test results.

## Running

Built for **Google Colab (T4 GPU)**, including through the Colab VS Code extension.

1. Put the datasets on Google Drive at `MyDrive/Colab Schoolwork/CAHSI 2026/Data/{VRIC, VeRi}`, or change `DRIVE_BASE` in the CONFIG cell.
2. Open `notebooks/ReIdTraining.ipynb` and click **Run All**.
   - The first run takes about 2 hours.
   - Checkpoints save to Drive every epoch, so a disconnect resumes where it left off.
   - Finished runs are cached.

The datasets are not included. Request them from their authors:
- **VRIC**: Kanacı et al., GCPR 2018
- **VeRi-776**: Liu et al., ECCV 2016

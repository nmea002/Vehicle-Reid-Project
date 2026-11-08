# Vehicle Re-Identification under Image Degradation

Vehicle re-ID experiments with [torchreid](https://github.com/KaiyangZhou/deep-person-reid) (OSNet x1_0, ImageNet-pretrained) on **VRIC** and **VeRi-776**, looking at how blur and low resolution affect retrieval accuracy.

## Layout

```
notebooks/ReIdTraining.ipynb   # all experiments (data subsets, training, k-fold CV, stratified eval)
data/                          # local dataset copies (git-ignored, not in this repo)
```

## Running

The notebook is meant for **Google Colab with a GPU** (works through the Colab VS Code extension).

1. Put the datasets on Google Drive:
   ```
   MyDrive/Spring 2026/CAHSI 2026/Data/
   ├── VRIC/   (train_images/, probe_images/, gallery_images/, vric_*.txt)
   └── VeRi/   (image_train/, image_query/, image_test/, ...)
   ```
2. Open `notebooks/ReIdTraining.ipynb` on a GPU runtime.
3. Run the install cell, mount Drive, and if your Drive folder is somewhere else, change `DRIVE_BASE` in the **CONFIG** cell.
4. Run the remaining cells in order. Checkpoints and logs are written to `DRIVE_BASE/logs` so they survive disconnects.

## Datasets

The datasets are not redistributed here. Request them from their authors:
- **VRIC**: Kanacı et al., *Vehicle Re-Identification in Context* (GCPR 2018)
- **VeRi-776**: Liu et al., *A Deep Learning-Based Approach to Progressive Vehicle Re-identification for Urban Surveillance* (ECCV 2016)

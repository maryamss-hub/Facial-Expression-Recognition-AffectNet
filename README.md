# Facial-Expression-Recognition-AffectNet
Facial Expression Recognition and Valence-Arousal Regression using CNN baselines (MobileNetV2, EfficientNetB0) with transfer learning and comparison of performance metrics.


##  Overview
This repository contains the implementation for **Assignment 01 - Deep Learning (CS452)** 
The task is **facial expression recognition (8 classes)** and **valence-arousal regression** using CNN baselines.

We compare:
- **MobileNetV2** (lightweight backbone with inverted residuals)
- **EfficientNetB0** (scaling-efficient CNN)
- **Custom CNN** 

---

##  Dataset
- Dataset provided by instructors (subset of AffectNet).
- 3999 cropped face images, resized to 224×224.
- Each image includes:
  - Expression label (0–7)
  - Valence ∈ [-1, +1]
  - Arousal ∈ [-1, +1]

---

##  Methods
- Preprocessing with `tf.data` pipeline (augmentation: flips, brightness, contrast, crops).
- Dual-head network: classification (softmax) + regression (tanh).
- Transfer learning: pretrained on ImageNet, fine-tuned on assignment dataset.

---

##  Results
| Model        | Accuracy | F1  | Kappa | Alpha | AUC   | PR-AUC | RMSE  | Corr  | SAGR  | CCC   |
|--------------|----------|-----|-------|-------|-------|--------|-------|-------|-------|-------|
| MobileNetV2  | 44.75%   | 43.98% | 0.369 | 0.366 | 0.831 | 0.457  | 0.370 | 0.514 | 0.598 | 0.490 |
| EfficientNetB0 | 44.25% | 44.28% | 0.363 | 0.363 | 0.829 | 0.454  | 0.359 | 0.533 | 0.643 | 0.486 |

 Training curves, confusion matrices, and sample predictions are available in `outputs/`.


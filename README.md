# Computer Vision Lab

A collection of classic computer-vision programs implemented as self-contained Jupyter notebooks.
Each notebook downloads any required data automatically — no local dataset setup is needed for the demos.

---

## Table of Contents

| #   | Topic                                                                                                                             | Notebook                                                                                 | Key Techniques                                                |
| --- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1   | [SIFT Image Matching](#1-sift-image-matching)                                                                                     | [`Sift/sift_matching.ipynb`](Sift/sift_matching.ipynb)                                   | SIFT · FLANN · Lowe Ratio · Homography (RANSAC)               |
| 2   | [Pedestrian Detection — HOG + SVM](#2-pedestrian-detection--hog--svm)                                                             | [`HOG/pedestrian_detection_hog_svm.ipynb`](HOG/pedestrian_detection_hog_svm.ipynb)       | HOG descriptor · RBF-SVM · Grid Search · HuggingFace Datasets |
| 3   | [Shape Similarity — Geometric Moments](#3-shape-similarity--geometric-moments)                                                    | [`Geometric_moments/geometric_moments.ipynb`](Geometric_moments/geometric_moments.ipynb) | Hu Moments · `cv2.matchShapes` · Affine Invariance            |
| 4   | [Block Matching — Motion Estimation (SAD)](#4-block-matching--motion-estimation-sad)                                              | [`block-matching/block_matching.ipynb`](block-matching/block_matching.ipynb)             | SAD · Exhaustive Block Search · Motion Vector Field           |
| 5   | [2D Image Registration · Affine Transforms · Object Recognition](#5-2d-image-registration--affine-transforms--object-recognition) | [`feature-matching/feature_matching.ipynb`](feature-matching/feature_matching.ipynb)     | SIFT · RANSAC Affine · Homography · Perspective Transform     |

---

## Programs

### 1 · SIFT Image Matching

**Notebook:** [`Sift/sift_matching.ipynb`](Sift/sift_matching.ipynb)

Implements SIFT-based feature matching between two images and visualises the matched keypoints.

**Steps:**

1. Download the Oxford _graf_ benchmark image pair automatically.
2. Detect keypoints and compute 128-D SIFT descriptors.
3. Match descriptors with FLANN + Lowe's ratio test.
4. Estimate a homography with RANSAC; distinguish inliers from outliers.
5. Visualise matches and project Image 1's boundary into Image 2.

---

### 2 · Pedestrian Detection — HOG + SVM

**Notebook:** [`HOG/pedestrian_detection_hog_svm.ipynb`](HOG/pedestrian_detection_hog_svm.ipynb)

Implements a pedestrian detection system using HOG features and an SVM classifier, and evaluates accuracy.

**Dataset:** [INRIA Person](https://huggingface.co/datasets/marcelarosalesj/inria-person) — pre-cropped patches loaded via HuggingFace `datasets` (~239 MB, auto-cached).

| Label | Class            | Description                           |
| ----- | ---------------- | ------------------------------------- |
| `1`   | `pedestrians`    | Cropped pedestrian patches (positive) |
| `0`   | `no_pedestrians` | Background patches (negative)         |

**Steps:**

1. Load the INRIA Person dataset from HuggingFace and preview a sample HOG descriptor.
2. Compute HOG features for all 2 487 patches; split 80 % train / 20 % test (stratified).
3. Scale features with `StandardScaler`; train an RBF-SVM via 3-fold grid search.
4. Evaluate with accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.
5. Visualise predicted labels on a random grid of test patches (green = correct, red = wrong).

---

### 3 · Shape Similarity — Geometric Moments

**Notebook:** [`Geometric_moments/geometric_moments.ipynb`](Geometric_moments/geometric_moments.ipynb)

Determines whether two shapes are similar by comparing their **Hu invariant moments**.

**Steps:**

1. Convert each image to a binary mask (white foreground).
2. Compute Hu moments with OpenCV; apply log transform for numerical stability.
3. Use `cv2.matchShapes` (Hu-moment distance) to score similarity.
4. Classify as _similar_ if the score is below a threshold.
5. Demo on synthetic shapes (circle vs. transformed circle vs. rectangle).

---

### 4 · Block Matching — Motion Estimation (SAD)

**Notebook:** [`block-matching/block_matching.ipynb`](block-matching/block_matching.ipynb)

Estimates motion between two consecutive video frames using block matching with **Sum of Absolute Differences (SAD)**.

$$
\text{SAD}(u,v) = \sum_{i=0}^{B-1}\sum_{j=0}^{B-1} |I_1(x+i,\, y+j) - I_2(x+u+i,\, y+v+j)|
$$

**Steps:**

1. Download the OpenCV `vtest.avi` sample video automatically.
2. Extract two frames separated by a small gap (≈ 5 frames).
3. Divide the reference frame into 16×16 blocks; exhaustively search a ±8 px window for the minimum-SAD match in the target frame.
4. Filter out flat (low-texture) blocks.
5. Visualise the SAD cost map and motion vector field.

---

### 5 · 2D Image Registration · Affine Transforms · Object Recognition

**Notebook:** [`feature-matching/feature_matching.ipynb`](feature-matching/feature_matching.ipynb)

Three tasks implemented with SIFT local features and OpenCV:

| Task                       | Goal                                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| **Image Registration**     | Align a _moving_ image to a _fixed_ reference using SIFT + RANSAC affine estimation.          |
| **Affine Transformations** | Apply and visualise translation, rotation, scale, shear, and combined transforms on an image. |
| **Object Recognition**     | Locate a known object template inside a cluttered scene using SIFT + RANSAC homography.       |

All images are downloaded automatically (OpenCV sample data).

---

## Requirements

Install dependencies with:

```bash
pip install opencv-python-headless numpy matplotlib scikit-learn scikit-image tqdm datasets Pillow
```

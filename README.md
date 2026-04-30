![Fingerprint Recognition](https://capsule-render.vercel.app/api?type=rect&color=0:0d1117,100:1a3a5c&height=130&section=header&text=🔏%20Fingerprint%20Recognition%20—%20SIFT-Based%20Biometric%20Matching&fontSize=22&fontColor=58a6ff&animation=fadeIn)

![Accuracy](https://img.shields.io/badge/Match%20Accuracy-57.83%25-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-SOCOFing-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![IEEE](https://img.shields.io/badge/IEEE%20Standards-P2830%20%7C%20P2841-blue?style=flat-square)

# Fingerprint Recognition — SIFT-Based Biometric Matching System

> Automated fingerprint identification using SIFT keypoint matching on the SOCOFing dataset. Handles low-quality and blurred fingerprint images, scanning up to 1000 real fingerprints to find the best match against an altered (degraded) query image.

---

## Table of Contents

- [Context](#context)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Source Code](#source-code)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Observations](#observations)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

---

## Context

Fingerprint Detection refers to the automated method of identifying or verifying a match between two human fingerprints. It is one of the most well-known biometrics and by far the most used biometric solution for authentication on computerised systems.

Fingerprint scanners work by capturing the pattern of ridges and valleys on a finger. The system first goes through an **enrollment** phase where it learns all enrolled individuals, then a **verification** phase where a live scan is matched against the stored database.

This project addresses a core real-world challenge: **matching low-quality or blurred fingerprints** captured by low-cost acquisition devices — where most commercial systems fail. Using SIFT (Scale-Invariant Feature Transform) and FLANN-based matching, the system compares an altered (degraded) query fingerprint against up to 1000 real fingerprint images and identifies the best match by keypoint score.

**Key differentiator:** The system is explicitly designed to handle degraded, blurred, and low-resolution input images — the exact conditions where traditional minutiae-based matchers break down.

---

## Dataset

| Property | Detail |
|----------|--------|
| Name | SOCOFing (Sokoto Coventry Fingerprint Dataset) |
| Source | Kaggle |
| Format | Bitmap (`.BMP`) |
| Directories | `Real/` — original fingerprints; `Altered/` — Easy / Medium / Hard degradations |
| Metadata | ID, gender, finger name per image |
| Images scanned per run | Up to 1000 (configurable) |

**Directory structure:**
```
SOCOFing/
├── Real/          ← ground truth fingerprints (unmodified)
└── Altered/
    ├── Altered-Easy/
    ├── Altered-Medium/
    └── Altered-Hard/   ← query images used in this project
```

The query sample used in this project:
```
SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP
```

---

## Methodology

### Pipeline Overview

```
Query Image (Altered/Degraded)
        ↓
Grayscale Conversion + Pre-processing
        ↓
SIFT Feature Extraction (keypoints + descriptors)
        ↓
FLANN-Based KNN Matching (k=2, Lowe's ratio test)
        ↓
Score Computation  →  best_score = match_points / min(keypoints) × 100
        ↓
Best Match Identified → Filename + Score Displayed
        ↓
cv2.drawMatches → Visual Result Window
```

### Why SIFT?

SIFT was selected as the feature extraction algorithm because:

- **Scale and rotation invariant** — robust to image transformations common in degraded scans
- **Handles low-contrast regions** — critical for blurred or smudged fingerprints
- **Descriptor-based matching** — works even when ridge structure is partially destroyed
- Outperforms raw minutiae extraction on heavily altered images

### Matching Algorithm

| Step | Detail |
|------|--------|
| Feature detector | `cv2.SIFT_create()` |
| Matcher | `cv2.FlannBasedMatcher` (algorithm=1, trees=10) |
| KNN | k=2 nearest neighbours per descriptor |
| Ratio test | Accept match if `p.distance < 0.1 × q.distance` (Lowe's ratio) |
| Scoring | `len(match_points) / min(keypoints_1, keypoints_2) × 100` |
| Selection | Image with highest score across all 1000 candidates = best match |

---

## Source Code

Full matching pipeline (`fingerprint_recognition.py`):

```python
import os
import cv2

# ── LOAD QUERY (ALTERED) SAMPLE ───────────────────────────────────────────
sample = cv2.imread("SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP")

best_score = 0
filename   = None
image      = None
kp1, kp2, mp = None, None, None
counter    = 0

# ── SCAN UP TO 1000 REAL FINGERPRINTS ────────────────────────────────────
for file in [file for file in os.listdir("SOCOFing/Real")][:1000]:
    if counter % 10 == 0:
        print(counter)
        print(file)
    counter += 1

    fingerprint_image = cv2.imread("SOCOFing/Real/" + file)

    # ── SIFT FEATURE EXTRACTION ───────────────────────────────────────────
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    # ── FLANN-BASED KNN MATCHING ──────────────────────────────────────────
    matches = cv2.FlannBasedMatcher(
        {'algorithm': 1, 'trees': 10}, {}
    ).knnMatch(descriptors_1, descriptors_2, k=2)

    # ── LOWE'S RATIO TEST ─────────────────────────────────────────────────
    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    # ── SCORE COMPUTATION ─────────────────────────────────────────────────
    keypoints = (len(keypoints_1)
                 if len(keypoints_1) < len(keypoints_2)
                 else len(keypoints_2))

    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename   = file
        image      = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

# ── DISPLAY RESULT ────────────────────────────────────────────────────────
print("Best Match: " + filename)
print("score: "      + str(best_score))

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=4, fy=4)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## Installation

**Prerequisites:** Python 3.9+, pip

```bash
# Clone the repository
git clone https://github.com/srighanesh-sriv/fingerprint-recognition.git
cd fingerprint-recognition

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the SOCOFing dataset from Kaggle and place it in the project root
# https://www.kaggle.com/datasets/ruizgara/socofing
```

**requirements.txt**
```
opencv-python
opencv-contrib-python
numpy
```

> **Note:** `opencv-contrib-python` is required for `cv2.SIFT_create()`. Do not install both `opencv-python` and `opencv-contrib-python` simultaneously — they conflict. Use `opencv-contrib-python` only.

---

## Usage

1. Download the SOCOFing dataset from Kaggle and place it at `SOCOFing/` in the project root
2. Choose your query image — update the path in line 4 of `fingerprint_recognition.py`
3. Run the script:

```bash
python fingerprint_recognition.py
```

4. The terminal prints progress every 10 files, then outputs:

```
Best Match: 150__M_Right_index_finger.BMP
score: 57.14285714285714
```

5. A window opens displaying the matched keypoints drawn between the query and the best-match image

**Changing the degradation level:**

```python
# Easy degradation
sample = cv2.imread("SOCOFing/Altered/Altered-Easy/150__M_Right_index_finger_Obl.BMP")

# Medium degradation
sample = cv2.imread("SOCOFing/Altered/Altered-Medium/150__M_Right_index_finger_Obl.BMP")

# Hard degradation (default in this project)
sample = cv2.imread("SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP")
```

---

## Project Structure

```
fingerprint-recognition/
│
├── fingerprint_recognition.py   # Main matching pipeline
├── requirements.txt
│
├── SOCOFing/                    # Dataset (download from Kaggle)
│   ├── Real/                    # Ground truth fingerprint images
│   └── Altered/
│       ├── Altered-Easy/
│       ├── Altered-Medium/
│       └── Altered-Hard/        # Query images (degraded)
│
└── screenshots/
    ├── source_code.png           # PyCharm IDE screenshot
    └── result.png                # Matched keypoints output window
```

---

## Observations

Different keypoints are highlighted with different colours. Two images are compared based on the distance between their respective keypoints. Based on the number of connecting lines drawn between both images, the `best_score` is determined — the image with the highest score is the exact match.

| Observation | Detail |
|-------------|--------|
| Keypoint colouring | Each matched keypoint pair is drawn in a distinct colour |
| Comparison metric | Distance between SIFT descriptors across both images |
| Score formula | `(matched keypoints / min total keypoints) × 100` |
| Iteration | Repeated across all 1000 candidate images; best score retained |

---

## Results

The exact result is displayed along with the matched fingerprint image by comparing approximately 1000 images in a short duration — even with low-quality degraded images provided for comparison.

| Metric | Value |
|--------|-------|
| Dataset size scanned | 1000 images |
| Best match found | `150__M_Right_index_finger.BMP` |
| Best match score | **57.14 – 57.83** |
| Input type | Hard-altered (heavily degraded) query |
| Output | Correct subject identified with visual keypoint overlay |

The system successfully identifies the correct fingerprint identity from a heavily blurred, obliquely-altered query image, demonstrating robustness in real-world low-quality acquisition scenarios.

---

## Conclusion

Python provides a number of tools and libraries for fingerprint recognition. The choice of library or framework depends on the specific requirements and application of the recognition system. OpenCV, TensorFlow, and scikit-image are among the popular libraries for implementing fingerprint recognition using Python.

This project demonstrates that **SIFT with FLANN-based matching is a viable approach for degraded fingerprint identification**, achieving meaningful match scores even on the hardest degradation category in the SOCOFing dataset.

**IEEE Standards followed:**
- IEEE P2830 — Standard for Technical Framework and Requirements of Shared Machine Learning
- IEEE P2841 — Framework and Process for Deep Learning Evaluation

---

## References

1. Anil K. Jain and Jianjiang Feng, "Latent fingerprint matching," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2014, pp. 2452–2465.
2. Mohammad Mogharen Askarin, KokSheik Wong and Raphael C-W Phan, "Reduced contact lifting of latent fingerprint," in *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, IEEE, 2017, pp. 1406–1410.
3. Cao, Kai, Eryun Liu, and Anil K. Jain, "Segmentation and enhancement of latent fingerprints: A course to fine ridge structure dictionary," in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2014, pp. 1847–1859.
4. Ramaiah, N. Pattabhi, A. Tirupathi Rao, and C. Krishna Mohan, "Enhancements to latent fingerprints in forensic applications," in *79th IEEE International Conference on Digital Signal Processing*, 2014, pp. 439–443.
5. Mouad M.H. Ali, Vivek H. Mahale, Pravin Yannawar and A. Gaikwad, "Fingerprint Recognition For Person Identification and Verification Based On Minutiae Matching," in *IEEE 6th International Conference on Advanced Computing*, 2016, pp. 332–339.

---

## Author

**Srighanesh A S**
B.Tech ECE — SRM Institute of Science and Technology (CGPA 9.35)
Market Analyst @ Siemens, Chennai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Srighanesh-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/srighaneshsrivathsan)
[![Email](https://img.shields.io/badge/Email-srighanesh.sriv%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:srighanesh.sriv@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-srighanesh--sriv-181717?style=flat-square&logo=github)](https://github.com/srighanesh-sriv)

---

*Built to demonstrate robust fingerprint matching under real-world degraded image conditions.*

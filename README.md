# PRISM-CTG
Traditional deep learning models for CTG analysis are often trained on limited patient cohorts or narrowly curated datasets, which constraints their potential performance. In this study, we introduce the first Foundation Model: PRISM-CTG, pre-trained with a multi-view self-supervised learning framework, specifically designed for CTG domain that incorporates clinical and patient context during pretraining. 

## Overview
This repositary provide the code for pretraining PRISM-CTG with multi-view self-supervised learning framework. Researchers may use the CTU-UHB dataset for experimentation and pipeline validation. However, for meaningful domain representation, the model should be trained on large-scale CTG data. Please replace the dataset with your institution’s data for full-scale pretraining. 

## PRISM-CTG learns meaningful CTG representation
2D-PCA visualisation on Task 4 showed meaningful representation based on the encoder representation alone, without additional linear-probing on the downstream dataset. 

<p align="center">
  <img src="PCA.png" alt="PRISM-CTG learns meaningful CTG representation" width="400">
</p>

## Usage
### Pre-training Input

A single `.npz` file containing:

| Key              | Shape        | Description              |
|------------------|--------------|--------------------------|
| `fhr_segments`   | `[N, 1200]`  | Fetal heart rate signal  |
| `toco_segments`  | `[N, 1200]`  | Tocodynamometer signal   |
| `gest_age`       | `[N]`         | Gestational age          |
| `maternal_age`   | `[N]`         | Maternal age             |
| `time_to_birth`  | `[N]`         | Time to birth            |

### Linear Probing Input

A directory containing:

| File            | Shape           | Description                                           |
|-----------------|-----------------|-------------------------------------------------------|
| `X_train.npy`   | `[N, 2, 1200]`  | Training signals (channel 0 = FHR, channel 1 = TOCO) |
| `y_train.npy`   | `[N]`            | Training labels                                      |
| `X_test.npy`    | `[N, 2, 1200]`  | Test signals                                         |
| `y_test.npy`    | `[N]`            | Test labels                                          |

### Pre-training
```
python run_pretraining.py --data_path /YOUR_CTG_DATA/CTG.npz 
```

### Linear-probing
```
python run_linear_probing.py 
```

## Dataset
### Pretraining
**Oxford Maternal Databasse (OXMAT)**: Unavailable due to privacy and ethical reason. Individual requests for access may be considered on a case-by-case basis, subject to institutional approval. Please contact gabriel.jones@wrh.ox.ac.uk.

**SPAM**: https://users.ox.ac.uk/~ndog0178/CTGchallenge2017.html

### Evaluation Dataset
**Oxmat-2025**: Unavailable due to privacy and ethical reason. Individual requests for access may be considered on a case-by-case basis, subject to institutional approval. Please contact gabriel.jones@wrh.ox.ac.uk.

**CTU-UHB**: https://archive.physionet.org/pn3/ctu-uhb-ctgdb/HEADER.shtml

**APHP-CTG**: The APHP-CTG dataset is expected to be made publicly available in 2026, subject to final administrative approval. A link to the official database will be added upon release.

## Notes
The original research code has been cleaned and simplified to improve readability and reduce implementation complexity. The complete version of the research code can be made available upon request.

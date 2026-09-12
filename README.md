# Alzheimer's Disease Classification with Machine Learning

Machine-learning project based on the **OASIS-2 longitudinal MRI dataset**. The goal is to distinguish visits from subjects classified as **Demented** and **Nondemented** using demographic, cognitive and MRI-derived features.

This repository was originally developed as a university Data Mining project and has been refactored into a reproducible portfolio project.

## Project goals

- clean and prepare longitudinal clinical data;
- compare multiple supervised-learning models;
- evaluate performance with subject-aware cross-validation;
- avoid leakage between repeated visits from the same person;
- interpret model performance with classification metrics and confusion matrices.

## Models

- Decision Tree
- Logistic Regression
- Support Vector Machine (RBF kernel)

## Features

The current models use:

- Age
- EDUC (years of education)
- SES (socioeconomic status)
- MMSE (Mini-Mental State Examination)
- eTIV (estimated total intracranial volume)
- nWBV (normalized whole-brain volume)
- ASF (atlas scaling factor)

`CDR` is intentionally **not used as a predictor** because it is a direct clinical measure of dementia severity and could make the classification problem unrealistically easy.

## Longitudinal validation

The OASIS-2 dataset contains repeated visits for the same subjects. A standard row-wise split can therefore place one visit from a subject in training and another visit from the same subject in validation.

To reduce this leakage risk, the refactored code uses **Leave-One-Group-Out cross-validation**, where `Subject ID` is the grouping variable. All visits from the held-out subject remain outside the training set for that fold.

## Treatment of the `Converted` group

Subjects labelled `Converted` transition from nondemented to demented during follow-up. Treating every visit from those subjects as demented would incorrectly assign their future status to earlier visits.

For this primary **current-status binary classification**, `Converted` observations are therefore excluded. A future extension could instead use baseline visits from converted subjects to build a true conversion-risk prediction task.

## Results

The refactored pipeline was executed in a clean Python 3.12 environment before the OASIS data file was removed from the public repository. The subject-aware cross-validation results were:

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.798 | 0.782 | 0.774 | 0.701 | 0.736 |
| Logistic Regression | **0.833** | **0.820** | **0.814** | **0.756** | **0.784** |
| SVM (RBF) | 0.795 | 0.774 | 0.787 | 0.669 | 0.723 |

Among the three baseline models, **Logistic Regression performs best across all reported metrics** under this validation setup.

These values are educational project results, not clinical-performance claims. The dataset is relatively small and the analysis is not intended to be used as a diagnostic system.

## Repository structure

```text
Data-Mining-Project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── presentation/
│   └── alzheimer_prediction_presentation.pdf
├── tests/
│   └── test_data_utils.py
├── .github/
│   └── workflows/
│       └── ci.yml
└── src/
    ├── data_utils.py
    ├── decision_tree.py
    ├── logistic_regression.py
    ├── svm.py
    └── model_comparison.py
```

The original university presentation is kept in **Italian**. The README is in English so the project can be reviewed quickly in an international context without rewriting the original academic material.

## How to run

1. Follow `data/README.md` to obtain OASIS-2 data under the applicable Data Use Agreement.
2. Save the demographic/longitudinal CSV as:

```text
data/oasis_longitudinal.csv
```

3. Install dependencies and run the comparison:

```bash
pip install -r requirements.txt
python src/model_comparison.py
```

The code uses paths relative to the repository and therefore does not depend on a specific local Windows username or folder.

## Evaluation metrics

The comparison script reports:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score

For a health-related classification task, accuracy alone can be misleading; recall and class-balanced metrics help describe performance on the positive class.

## Reproducibility and tests

The public repository does not contain the OASIS data file. The GitHub Actions workflow therefore runs **dataset-independent unit tests** that validate the preprocessing logic. After an authorized user adds the dataset locally, the full model comparison can be reproduced with the command above.

## Data source and acknowledgement

This project uses **OASIS-2: Longitudinal MRI Data in Nondemented and Demented Older Adults**. OASIS requires acceptance of its Data Use Agreement and acknowledgement of the data when results derived from it are publicly presented.

Relevant publication:

Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010). *Open Access Series of Imaging Studies: Longitudinal MRI Data in Nondemented and Demented Older Adults*. Journal of Cognitive Neuroscience, 22(12), 2677–2684. https://doi.org/10.1162/jocn.2009.21407

## Future improvements

- investigate prediction of future conversion using baseline visits;
- add hyperparameter tuning with nested/group-aware validation;
- add ROC-AUC / PR-AUC using out-of-fold scores;
- improve feature engineering and missing-data handling;
- expand automated unit tests.

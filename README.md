# Alzheimer's Disease Classification with Machine Learning

Machine-learning project based on the **OASIS longitudinal MRI dataset**. The goal is to distinguish visits from subjects classified as **Demented** and **Nondemented** using demographic, cognitive and MRI-derived features.

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

The OASIS dataset contains repeated visits for the same subjects. A standard row-wise split can therefore place one visit from a subject in training and another visit from the same subject in validation.

To reduce this leakage risk, the refactored code uses **Leave-One-Group-Out cross-validation**, where `Subject ID` is the grouping variable. All visits from the held-out subject remain outside the training set for that fold.

## Treatment of the `Converted` group

Subjects labelled `Converted` transition from nondemented to demented during follow-up. Treating every visit from those subjects as demented would incorrectly assign their future status to earlier visits.

For this primary **current-status binary classification**, `Converted` observations are therefore excluded. A separate future extension could use baseline visits from converted subjects to build a true conversion-risk prediction task.

## Repository structure

```text
Data-Mining-Project/
├── README.md
├── requirements.txt
├── .gitignore
├── oasis_longitudinal.csv
├── Presentazione (1).pdf
└── src/
    ├── data_utils.py
    ├── decision_tree.py
    ├── logistic_regression.py
    ├── svm.py
    └── model_comparison.py
```

## How to run

Create a virtual environment, install the dependencies and run any script from the repository root:

```bash
pip install -r requirements.txt
python src/model_comparison.py
```

The code locates the dataset with a path relative to the repository, so it does not depend on a specific local Windows username or folder.

## Evaluation metrics

The comparison script reports:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score

For a health-related classification task, accuracy alone can be misleading; recall and class-balanced metrics are useful for understanding errors on the positive class.

## Data source

The project uses the OASIS longitudinal dataset. If reusing or redistributing the data, consult the official OASIS documentation and terms of use and provide the required attribution.

## Future improvements

- investigate prediction of future conversion using baseline visits;
- add hyperparameter tuning with nested/group-aware validation;
- add ROC-AUC / PR-AUC using out-of-fold scores;
- improve feature engineering and missing-data handling;
- add automated tests and continuous integration.

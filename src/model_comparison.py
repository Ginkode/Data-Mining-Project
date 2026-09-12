"""Compare the main classifiers using subject-aware cross-validation."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utils import load_classification_data


def evaluate_model(name, model, X, y, groups, cv):
    predictions = cross_val_predict(model, X, y, groups=groups, cv=cv)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
    }


def main():
    _, X, y, groups = load_classification_data()
    cv = LeaveOneGroupOut()

    models = {
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced"),
        ),
    }

    results = [
        evaluate_model(name, model, X, y, groups, cv)
        for name, model in models.items()
    ]

    results_df = pd.DataFrame(results).set_index("Model")
    print("\n=== SUBJECT-AWARE CROSS-VALIDATION RESULTS ===")
    print(results_df.round(3).to_string())


if __name__ == "__main__":
    main()

"""Logistic-regression analysis with subject-aware cross-validation."""

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import load_classification_data


def main():
    _, X, y, groups = load_classification_data()

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )

    cv = LeaveOneGroupOut()
    predictions = cross_val_predict(model, X, y, groups=groups, cv=cv)

    print("\n=== LOGISTIC REGRESSION: SUBJECT-AWARE CV ===")
    print(classification_report(y, predictions, target_names=["Nondemented", "Demented"]))

    ConfusionMatrixDisplay.from_predictions(
        y,
        predictions,
        display_labels=["Nondemented", "Demented"],
        cmap="Blues",
    )
    plt.title("Logistic Regression — Leave-One-Subject-Out CV")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

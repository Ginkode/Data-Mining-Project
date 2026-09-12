"""Support-vector-machine analysis with subject-aware cross-validation."""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utils import load_classification_data


def main():
    _, X, y, groups = load_classification_data()

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced"),
    )

    cv = LeaveOneGroupOut()
    predictions = cross_val_predict(model, X, y, groups=groups, cv=cv)

    print("\n=== SVM (RBF): SUBJECT-AWARE CV ===")
    print(classification_report(y, predictions, target_names=["Nondemented", "Demented"]))

    ConfusionMatrixDisplay.from_predictions(
        y,
        predictions,
        display_labels=["Nondemented", "Demented"],
        cmap="Purples",
    )
    plt.title("SVM (RBF) — Leave-One-Subject-Out CV")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

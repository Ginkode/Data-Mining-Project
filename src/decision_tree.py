"""Decision-tree analysis with subject-aware cross-validation."""

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from data_utils import FEATURES, load_classification_data


def build_model():
    return tree.DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )


def main():
    _, X, y, groups = load_classification_data()
    model = build_model()
    cv = LeaveOneGroupOut()
    predictions = cross_val_predict(model, X, y, groups=groups, cv=cv)

    print("\n=== DECISION TREE: SUBJECT-AWARE CV ===")
    print(classification_report(y, predictions, target_names=["Nondemented", "Demented"]))

    ConfusionMatrixDisplay.from_predictions(
        y,
        predictions,
        display_labels=["Nondemented", "Demented"],
        cmap="Blues",
    )
    plt.title("Decision Tree — Leave-One-Subject-Out CV")
    plt.tight_layout()
    plt.show()

    model.fit(X, y)
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        model,
        feature_names=FEATURES,
        class_names=["Nondemented", "Demented"],
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Decision Tree fitted on the full analysis dataset")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

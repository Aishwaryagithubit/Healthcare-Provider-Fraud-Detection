import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def get_probabilities(model, X):

    return model.predict_proba(X)[:, 1]


def evaluate_model(model, X, y, threshold=0.50):

    probabilities = get_probabilities(model, X)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = {
        "Accuracy": accuracy_score(
            y,
            predictions
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y,
            predictions
        ),

        "Precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y,
            probabilities
        ),

        "PR-AUC": average_precision_score(
            y,
            probabilities
        )
    }

    return results


def find_best_threshold(model, X_valid, y_valid):

    probabilities = get_probabilities(
        model,
        X_valid
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(
        0.10,
        0.91,
        0.01
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_valid,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def confusion_matrix_at_threshold(
    model,
    X,
    y,
    threshold
):

    probabilities = get_probabilities(
        model,
        X
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return confusion_matrix(
        y,
        predictions
    )
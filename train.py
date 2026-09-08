from pathlib import Path
import json
import pickle

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline


from data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    load_data,
    prepare_data,
    split_data
)

from model_evaluation import (
    evaluate_model,
    find_best_threshold
)


RANDOM_STATE = 42

DATA_FILE = "healthcare_fraud_detection.csv"

MODEL_DIR = Path("model")

MODEL_FILE = (
    MODEL_DIR /
    "healthcare_fraud_model.pkl"
)


def build_logistic_pipeline():

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),

        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE
            )
        )
    ])


def build_random_forest_pipeline():

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),

        (
            "model",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])


def build_gradient_boosting_pipeline():

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),

        (
            "model",
            GradientBoostingClassifier(
                random_state=RANDOM_STATE
            )
        )
    ])


def tune_models(X_train, y_train):

    # Logistic Regression

    logistic_grid = GridSearchCV(

        build_logistic_pipeline(),

        {
            "model__C": [
                0.01,
                0.1,
                1,
                10
            ],

            "model__solver": [
                "liblinear",
                "lbfgs"
            ]
        },

        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    logistic_grid.fit(
        X_train,
        y_train
    )


    # Random Forest

    rf_grid = GridSearchCV(

        build_random_forest_pipeline(),

        {
            "model__max_depth": [
                None,
                10,
                20
            ],

            "model__min_samples_leaf": [
                1,
                2
            ],

            "model__min_samples_split": [
                2,
                5
            ],

            "model__n_estimators": [
                200,
                300
            ]
        },

        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    rf_grid.fit(
        X_train,
        y_train
    )


    # Gradient Boosting

    gradient_grid = GridSearchCV(

        build_gradient_boosting_pipeline(),

        {
            "model__learning_rate": [
                0.05,
                0.10
            ],

            "model__max_depth": [
                2,
                3
            ],

            "model__min_samples_leaf": [
                1,
                2
            ],

            "model__n_estimators": [
                100,
                200
            ]
        },

        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    gradient_grid.fit(
        X_train,
        y_train
    )


    print("\nBest Logistic Regression:")
    print(logistic_grid.best_params_)
    print(
        "CV F1:",
        round(
            logistic_grid.best_score_,
            4
        )
    )


    print("\nBest Random Forest:")
    print(rf_grid.best_params_)
    print(
        "CV F1:",
        round(
            rf_grid.best_score_,
            4
        )
    )


    print("\nBest Gradient Boosting:")
    print(gradient_grid.best_params_)
    print(
        "CV F1:",
        round(
            gradient_grid.best_score_,
            4
        )
    )


    return (
        logistic_grid.best_estimator_,
        rf_grid.best_estimator_,
        gradient_grid.best_estimator_
    )


def main():

    print("Loading dataset...")

    df = load_data(
        DATA_FILE
    )

    print(
        "Dataset shape:",
        df.shape
    )


    X, y = prepare_data(df)

    print(
        "Features:",
        X.shape
    )

    print("\nFraud distribution:")

    print(
        y.value_counts()
    )


    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    ) = split_data(
        X,
        y,
        random_state=RANDOM_STATE
    )


    print(
        "\nTrain:",
        X_train.shape
    )

    print(
        "Validation:",
        X_valid.shape
    )

    print(
        "Test:",
        X_test.shape
    )


    (
        logistic_model,
        random_forest_model,
        gradient_model
    ) = tune_models(
        X_train,
        y_train
    )


    # Final ensemble

    ensemble = VotingClassifier(

        estimators=[

            (
                "logistic",
                logistic_model
            ),

            (
                "random_forest",
                random_forest_model
            ),

            (
                "gradient_boosting",
                gradient_model
            )
        ],

        voting="soft",

        weights=[
            1,
            2,
            2
        ]
    )


    print(
        "\nTraining final ensemble..."
    )

    ensemble.fit(
        X_train,
        y_train
    )


    # Validation

    print(
        "\nValidation performance:"
    )

    validation_scores = evaluate_model(
        ensemble,
        X_valid,
        y_valid
    )

    for metric, value in validation_scores.items():

        print(
            f"{metric}: {value:.4f}"
        )


    # Threshold optimisation

    (
        best_threshold,
        best_validation_f1
    ) = find_best_threshold(
        ensemble,
        X_valid,
        y_valid
    )


    print(
        "\nBest validation threshold:",
        round(best_threshold, 2)
    )

    print(
        "Best validation F1:",
        round(best_validation_f1, 4)
    )


    # Final test evaluation

    print(
        "\nFinal test performance:"
    )

    test_scores = evaluate_model(
        ensemble,
        X_test,
        y_test,
        threshold=best_threshold
    )

    for metric, value in test_scores.items():

        print(
            f"{metric}: {value:.4f}"
        )


    # Save model

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    model_package = {

        "model": ensemble,

        "threshold": best_threshold,

        "feature_columns": list(
            X.columns
        ),

        "categorical_features":
            CATEGORICAL_FEATURES,

        "numerical_features":
            NUMERICAL_FEATURES,

        "target":
            "Is_Fraud",

        "test_metrics":
            test_scores
    }


    with open(
        MODEL_FILE,
        "wb"
    ) as file:

        pickle.dump(
            model_package,
            file
        )


    with open(
        MODEL_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            test_scores,
            file,
            indent=4
        )


    print(
        "\nModel saved to:"
    )

    print(
        MODEL_FILE
    )

    print(
        "\nTraining completed successfully."
    )


if __name__ == "__main__":
    main()
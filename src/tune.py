import hydra
import optuna
import dagshub
import mlflow
import mlflow.sklearn

import pandas as pd

from omegaconf import DictConfig

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================
# DagsHub + MLflow
# =========================
dagshub.init(
    repo_owner="NullBitZer0",
    repo_name="titanic-survival",
    mlflow=True
)

mlflow.set_experiment("Optuna-Hydra-MLflow")


# =========================
# Hydra Main
# =========================
@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
)
def main(cfg: DictConfig):

    # =========================
    # Load Data
    # =========================
    df = pd.read_csv("processed/processed.csv")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state
    )

    # =========================
    # Objective Function
    # =========================
    def objective(trial):

        # Optuna search space
        n_estimators = trial.suggest_int(
            "n_estimators",
            50,
            500
        )

        max_depth = trial.suggest_int(
            "max_depth",
            2,
            20
        )

        # MLflow tracking
        with mlflow.start_run(nested=True):

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            accuracy = accuracy_score(y_test, preds)

            # Log params
            mlflow.log_param(
                "n_estimators",
                n_estimators
            )

            mlflow.log_param(
                "max_depth",
                max_depth
            )

            # Log metric
            mlflow.log_metric(
                "accuracy",
                accuracy
            )

            return accuracy

    # =========================
    # Parent Run
    # =========================
    with mlflow.start_run(run_name="optuna_optimization"):

        study = optuna.create_study(
            direction="maximize"
        )

        study.optimize(
            objective,
            n_trials=20
        )

        print("\nBest Trial")
        print("Accuracy:", study.best_value)
        print("Params:", study.best_params)


if __name__ == "__main__":
    main()
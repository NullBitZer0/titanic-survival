import json
import yaml
import dagshub
import mlflow
import mlflow.sklearn
import hydra
from omegaconf import DictConfig
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
)
def main(cfg: DictConfig):

    n_estimators = cfg.model.n_estimators
    max_depth = cfg.model.max_depth
    test_size = cfg.data.test_size

    print(cfg)

# =========================
# DagsHub + MLflow
# =========================
dagshub.init(
    repo_owner="NullBitZer0",
    repo_name="titanic-survival",
    mlflow=True
)

mlflow.set_experiment("Titanic DVC Pipeline")

mlflow.sklearn.autolog()


# =========================
# Load Params
# =========================
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

n_estimators = params["model"]["n_estimators"]
max_depth = params["model"]["max_depth"]
test_size = params["data"]["test_size"]

# =========================
# Load Data
# =========================

df = pd.read_csv("processed/processed.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

# =========================
# Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

# =========================
# Train
# =========================
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    mlflow.sklearn.log_model(
     sk_model=model,
     name="random_forest_model",
     registered_model_name="TitanicRandomForest"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    print(f"Accuracy: {accuracy}")

    # Save metrics
    metrics = {
        "accuracy": accuracy
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
if __name__ == "__main__":
    main()
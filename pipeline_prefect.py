"""
pipeline_prefect.py
Atelier 3 - Pipeline ML orchestre avec Prefect.

Utilisation :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow code
"""

import argparse
import subprocess
import sys

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from model_pipeline import (
    evaluate_model,
    load_model,
    predict,
    prepare_data,
    save_model,
    train_model,
)

MODEL_PATH = "churn_model.joblib"
DATA_PATH = "Churn_Modelling.csv"
# Les outils de qualite/securite ne scannent QUE ces fichiers
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run(cmd, stop_on_error=True):
    """Lance une commande Python (-m) et affiche son resultat."""
    print(">>", " ".join(cmd))
    result = subprocess.run([sys.executable, "-m", *cmd], check=False)
    if stop_on_error and result.returncode != 0:
        raise RuntimeError(f"Echec de la commande : {' '.join(cmd)}")


# ---------------- Taches liees au code ----------------


@task(name="install")
def install_dependencies():
    run(["pip", "install", "-r", "requirements.txt"])


@task(name="format")
def format_code():
    run(["black", *FILES])


@task(name="quality")
def quality_code():
    run(["flake8", "--max-line-length=88", *FILES], stop_on_error=False)


@task(name="security")
def security_code():
    run(["bandit", *FILES], stop_on_error=False)


@task(name="tests")
def unit_tests():
    run(["pytest", "tests", "-v"])


# ---------------- Taches liees aux donnees / modele ----------------


@task(name="prepare", cache_policy=NO_CACHE)
def prepare_task(data_path=DATA_PATH):
    return prepare_data(data_path)


@task(name="train", cache_policy=NO_CACHE)
def train_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="save", cache_policy=NO_CACHE)
def save_task(model, scaler, model_path=MODEL_PATH):
    save_model(model, scaler, model_path)


@task(name="load", cache_policy=NO_CACHE)
def load_task(model_path=MODEL_PATH):
    return load_model(model_path)


@task(name="evaluate", cache_policy=NO_CACHE)
def evaluate_task(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


@task(name="predict", cache_policy=NO_CACHE)
def predict_task(model, scaler):
    sample = [[850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]]
    resultat = predict(model, scaler, sample)
    print(f"Prediction : {resultat[0]} (0 = reste, 1 = churn)")
    return resultat


# ---------------- Flows ----------------


@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    unit_tests()


@flow(name="train")
def train_flow():
    X_train, X_test, y_train, y_test, scaler = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model, scaler)


@flow(name="evaluate")
def evaluate_flow():
    X_train, X_test, y_train, y_test, scaler = prepare_task()
    model, scaler = load_task()
    evaluate_task(model, X_test, y_test)


@flow(name="all")
def all_flow():
    # Etapes liees au code
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    unit_tests()
    # Etapes liees aux donnees / modele
    X_train, X_test, y_train, y_test, scaler = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model, scaler)
    evaluate_task(model, X_test, y_test)
    predict_task(model, scaler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument(
        "--flow",
        required=True,
        choices=["all", "train", "entrainement", "evaluate", "code"],
    )
    args = parser.parse_args()

    flows = {
        "all": all_flow,
        "train": train_flow,
        "entrainement": train_flow,
        "evaluate": evaluate_flow,
        "code": code_flow,
    }
    flows[args.flow]()

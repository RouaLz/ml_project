"""
model_pipeline.py
-----------------
Fonctions modularisées du pipeline ML de prédiction du churn bancaire.
Issues de la modularisation du notebook customer_churn.ipynb (Atelier 2 - MLOps).
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Colonnes non informatives pour le modèle
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]
TARGET = "Exited"


def prepare_data(path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """Charge et prétraite les données du churn.

    Étapes :
      1. Lecture du CSV.
      2. Encodage de la variable catégorielle 'Gender'.
      3. Suppression des colonnes non informatives.
      4. Séparation features / target.
      5. Split train/test.
      6. Standardisation (fit sur le train uniquement).

    Args:
        path (str): chemin vers le fichier CSV.
        test_size (float): proportion du jeu de test.
        random_state (int): graine pour la reproductibilité.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    df = pd.read_csv(path)

    # Encodage de Gender (Female -> 0, Male -> 1)
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Suppression des colonnes inutiles
    df = df.drop(columns=COLUMNS_TO_DROP)

    # Séparation features / target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Standardisation : fit sur le train, transform sur les deux
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier sur les données d'entraînement.

    Args:
        X_train (array): features d'entraînement.
        y_train (array): labels d'entraînement.
        n_estimators (int): nombre d'arbres de la forêt.
        random_state (int): graine pour la reproductibilité.

    Returns:
        RandomForestClassifier: le modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue les performances du modèle sur le jeu de test.

    Affiche l'accuracy, la matrice de confusion et le rapport de classification.

    Args:
        model: modèle entraîné exposant une méthode predict().
        X_test (array): features de test.
        y_test (array): labels réels de test.

    Returns:
        dict: métriques {'accuracy', 'confusion_matrix', 'classification_report'}.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy score : {accuracy * 100:.2f} %")
    print("\nMatrice de confusion :")
    print(matrix)
    print("\nRapport de classification :")
    print(report)

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "classification_report": report,
    }


def save_model(model, scaler=None, path="churn_model.joblib"):
    """Sauvegarde le modèle entraîné (et le scaler) avec joblib.

    Le scaler est sauvegardé avec le modèle : sans lui, toute nouvelle
    prédiction sur des données brutes serait incorrecte.

    Args:
        model: modèle entraîné.
        scaler: StandardScaler ajusté (optionnel).
        path (str): chemin du fichier de sortie.

    Returns:
        str: le chemin du fichier sauvegardé.
    """
    joblib.dump({"model": model, "scaler": scaler}, path)
    print(f"Modèle sauvegardé dans : {path}")
    return path


def load_model(path="churn_model.joblib"):
    """Charge un modèle sauvegardé.

    Args:
        path (str): chemin du fichier .joblib.

    Returns:
        tuple: (model, scaler)
    """
    artefacts = joblib.load(path)
    print(f"Modèle chargé depuis : {path}")
    return artefacts["model"], artefacts["scaler"]


def predict(model, scaler, sample):
    """Prédit le churn pour un ou plusieurs clients.

    Args:
        model: modèle entraîné.
        scaler: StandardScaler ajusté lors du prepare_data.
        sample (list | array): une ou plusieurs lignes de features brutes,
            dans l'ordre : CreditScore, Gender, Age, Tenure, Balance,
            NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.

    Returns:
        array: prédictions (0 = reste, 1 = churn).
    """
    if not isinstance(sample, pd.DataFrame):
        colonnes = getattr(scaler, "feature_names_in_", None)
        sample = pd.DataFrame(sample, columns=colonnes)
    if scaler is not None:
        sample = scaler.transform(sample)
    return model.predict(sample)

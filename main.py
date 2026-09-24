"""
main.py
-------
Point d'entrée du pipeline ML : exécute les différentes étapes via des arguments CLI.
Atelier 2 - Modularisation du code (MLOps).

Exemples d'utilisation :
    python main.py --prepare
    python main.py --train --save
    python main.py --evaluate
    python main.py --all
"""

import argparse

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


def parse_args():
    """Définit et récupère les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Pipeline ML - prédiction du churn bancaire"
    )
    parser.add_argument(
        "--data", type=str, default=DATA_PATH, help="Chemin du fichier CSV"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Chemin du fichier modèle"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="Préparer les données uniquement"
    )
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument(
        "--evaluate", action="store_true", help="Évaluer le modèle sauvegardé"
    )
    parser.add_argument("--save", action="store_true", help="Sauvegarder le modèle")
    parser.add_argument("--all", action="store_true", help="Exécuter tout le pipeline")
    return parser.parse_args()


def main():
    """Orchestre l'exécution des étapes du pipeline selon les arguments CLI."""
    args = parse_args()

    # Si aucune action n'est demandée, on exécute tout le pipeline
    if not any([args.prepare, args.train, args.evaluate, args.save, args.all]):
        args.all = True

    # --- Étape 1 : préparation des données ---
    print("=== Préparation des données ===")
    X_train, X_test, y_train, y_test, scaler = prepare_data(args.data)
    print(f"X_train : {X_train.shape} | X_test : {X_test.shape}")
    print(f"Répartition de la cible (train) :\n{y_train.value_counts()}\n")

    if args.prepare and not (args.train or args.evaluate or args.save or args.all):
        return

    model = None

    # --- Étape 2 : entraînement ---
    if args.train or args.all:
        print("=== Entraînement du modèle ===")
        model = train_model(X_train, y_train)
        print("Modèle entraîné.\n")

    # --- Étape 3 : sauvegarde ---
    if (args.save or args.all) and model is not None:
        print("=== Sauvegarde du modèle ===")
        save_model(model, scaler, args.model)
        print()

    # --- Étape 4 : évaluation ---
    if args.evaluate or args.all:
        print("=== Évaluation du modèle ===")
        if model is None:
            model, scaler = load_model(args.model)
        evaluate_model(model, X_test, y_test)
        print()

    # --- Étape 5 : exemple de prédiction ---
    if args.all and model is not None:
        print("=== Exemple de prédiction ===")
        # CreditScore, Gender, Age, Tenure, Balance,
        # NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
        sample = [[850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]]
        resultat = predict(model, scaler, sample)
        print(f"Prédiction : {resultat[0]} (0 = reste, 1 = churn)")


if __name__ == "__main__":
    main()

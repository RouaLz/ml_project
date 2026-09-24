"""
deploiement_prefect.py
Atelier 3 - Deploiement et planification des flows Prefect.
"""

from prefect import serve

from pipeline_prefect import all_flow, code_flow, evaluate_flow, train_flow

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",  # tous les jours a 2h du matin
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(
            name="ml-pipeline-train",
            tags=["training", "mlops"],
        ),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate",
            tags=["evaluation", "mlops"],
        ),
        code_flow.to_deployment(
            name="ml-pipeline-code",
            tags=["quality", "mlops"],
        ),
    )

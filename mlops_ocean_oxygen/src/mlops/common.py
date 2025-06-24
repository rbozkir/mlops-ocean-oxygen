from mlflow import MlflowClient
import os
import mlflow


def ensure_experiment_exists(experiment_name: str):

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not tracking_uri:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI environment variables must be set."
        )
    else:
        mlflow.set_tracking_uri(tracking_uri)

    if not username or not password:
        raise EnvironmentError(
            "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD environment variables must be set."
        )

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
        print(f"Experiment '{experiment_name}' created. ID: {experiment_id}")
    else:
        print(
            f"Experiment '{experiment_name}' already exists. ID: {experiment.experiment_id}"
        )
    
    return experiment

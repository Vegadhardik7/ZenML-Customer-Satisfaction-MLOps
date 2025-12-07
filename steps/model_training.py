import logging
import mlflow
from mlflow import sklearn as mlflow_sklearn
import pandas as pd
from zenml import step
from src.model_development import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name if experiment_tracker is not None else None)
def train_model(
    X_train: pd.DataFrame,  
    y_train: pd.Series, 
    config: ModelNameConfig,
    ) -> RegressorMixin:
    """
    Train model on the provided data.
    """
    try:
        if config.model_name == "LinearRegression":
            mlflow_sklearn.autolog()
            model = LinearRegressionModel()
            lr_model = model.train(X_train, y_train)
            return lr_model
        else:
            raise ValueError(f"Model {config.model_name} not listed.")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
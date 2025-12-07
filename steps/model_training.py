import logging
import pandas as pd
from zenml import step
from src.model_development import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,  
    y_train: pd.Series, 
    config: ModelNameConfig,
    ) -> RegressorMixin:
    """
    Train model on the provided data.
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            lr_model = model.train(X_train, y_train)
            return lr_model
        else:
            raise ValueError(f"Model {config.model_name} not listed.")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
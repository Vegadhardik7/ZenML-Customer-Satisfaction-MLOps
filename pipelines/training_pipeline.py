import logging
import pandas as pd
from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.cleaning_data import clean_data
from steps.model_training import train_model
from steps.config import ModelNameConfig
from steps.evaluate_model import evaluate_model

logger = logging.getLogger(__name__)

@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    """
    Training pipeline with customer satisfaction prediction.
    ZenML automatically integrates with MLflow for experiment tracking.
    """
    data = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(data)
    
    # Create config for the model
    config = ModelNameConfig()
    model = train_model(X_train, y_train, config)
    mse, r2, rmse = evaluate_model(model, X_test, y_test)
    
    return mse, r2, rmse

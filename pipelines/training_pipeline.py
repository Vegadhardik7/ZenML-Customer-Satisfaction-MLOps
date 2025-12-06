import logging
import pandas as pd
from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.cleaning_data import clean_data
from steps.model_training import train_model
from steps.evaluate_model import evaluate_model

@pipeline
def training_pipeline(data_path: str):
    data = ingest_data(data_path)
    df = clean_data(data)
    train_model(df)
    evaluate_model(df)

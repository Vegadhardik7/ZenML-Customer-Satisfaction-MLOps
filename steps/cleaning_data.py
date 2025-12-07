import logging
from typing import Tuple
import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataDivisionStrategy, DataPreprocessStrategy


@step
def clean_data(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    """
    ZenML step that cleans and splits the dataset.
    Ensures consistent return values for downstream steps.
    """
    try:
        logging.info("Starting data cleaning step.")

        preprocess = DataCleaning(df, DataPreprocessStrategy())
        processed_df = preprocess.handle_data()

        if not isinstance(processed_df, pd.DataFrame):
            raise TypeError("Expected DataFrame after preprocessing.")

        splitter = DataCleaning(processed_df, DataDivisionStrategy())
        result = splitter.handle_data()

        if not isinstance(result, tuple):
            raise TypeError("Split result must be a tuple of 4 elements.")

        X_train, X_test, y_train, y_test = result

        logging.info("Data cleaning completed successfully.")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error during cleaning step: {e}")
        raise

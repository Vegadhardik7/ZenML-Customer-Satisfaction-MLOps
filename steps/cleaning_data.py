import logging
import pandas as pd
from zenml import step

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Simple data cleaning step.

    Currently performs minimal cleaning and returns the cleaned DataFrame
    so downstream steps receive a proper `pd.DataFrame` instance.
    """
    try:
        # Example minimal cleaning: drop fully empty rows and reset index
        cleaned = df.dropna(how="all").reset_index(drop=True)
        return cleaned
    except Exception as e:
        logging.error(f"Error while cleaning data: {e}")
        raise
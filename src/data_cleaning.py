import logging
from typing import Union, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """Abstract base class that enforces a structure for data operations."""

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[
        pd.DataFrame,
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ]:
        """Subclasses must implement a data operation."""
        pass


class DataPreprocessStrategy(DataStrategy):
    """Strategy for preprocessing data."""

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info("Preprocessing data.")
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
                errors="ignore"
            )

            numerical_cols = [
                "product_weight_g", "product_length_cm",
                "product_height_cm", "product_width_cm"
            ]

            for col in numerical_cols:
                if col in data.columns:
                    data[col].fillna(data[col].median(), inplace=True)

            if "review_comment_message" in data.columns:
                data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])

            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            for col in cols_to_drop:
                if col in data.columns:
                    data = data.drop(col, axis=1)

            return data

        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise


class DataDivisionStrategy(DataStrategy):
    """Strategy for dividing data into train and test sets."""

    def handle_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        try:
            if "review_score" not in data.columns:
                raise ValueError("'review_score' column missing from dataset.")

            X = data.drop(['review_score'], axis=1)
            y = data['review_score']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error in train-test split: {e}")
            raise


class DataCleaning:
    """Orchestrates different cleaning strategies."""

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[
        pd.DataFrame,
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ]:
        return self.strategy.handle_data(self.data)

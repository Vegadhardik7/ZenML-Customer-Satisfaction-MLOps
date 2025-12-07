import logging
import pandas as pd
import numpy as np
from typing import Tuple, Protocol, runtime_checkable, Any
from typing_extensions import Annotated
from zenml import step
from src.model_evaluation import MSE, R2, RMSE

@runtime_checkable
class RegressorProtocol(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

@step
def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple [Annotated[float, "mse"], Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    """
    Evaluate model performance on the provided test data.

    Args:
        model: Trained regressor model with a `predict` method.
        X_test: Feature DataFrame for testing.
        y_test: True target values.

    Returns:
        float: Mean Squared Error (MSE) of the predictions.
    """
    pred = np.asarray(model.predict(X_test))
    mse_class = MSE()
    mse = mse_class.calculate_score(y_test.to_numpy(), pred)

    r2_class = R2()
    r2 = r2_class.calculate_score(y_test.to_numpy(), pred)

    rmse_class = RMSE()
    rmse = rmse_class.calculate_score(y_test.to_numpy(), pred)

    return mse, r2, rmse
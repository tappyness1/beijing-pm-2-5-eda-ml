from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from typing import Any, Tuple
import numpy as np
import pandas as pd

class ForecastModel:
    """Random Forest model for PM2.5 forecasting."""
    def __init__(self) -> None:
        """
        Initialize the Random Forest regressor.
        """
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the model to training data.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target values.
        """
        self.model.fit(X, y)

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[float, float]:
        """
        Compute train and test mean squared error (MSE).

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target values.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target values.

        Returns:
            Tuple[float, float]: Train MSE, Test MSE.
        """
        y_train_pred = self.predict(X_train)
        train_error = mean_squared_error(y_train, y_train_pred)

        y_test_pred = self.predict(X_test)
        test_error = mean_squared_error(y_test, y_test_pred)

        return train_error, test_error

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target values for input features.

        Args:
            X (pd.DataFrame): Features for prediction.

        Returns:
            np.ndarray: Predicted target values.
        """
        return self.model.predict(X)
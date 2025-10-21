from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class ForecastModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X_train, y_train, X_test, y_test):
        y_train_pred = self.predict(X_train)
        train_error = mean_squared_error(y_train, y_train_pred)

        y_test_pred = self.predict(X_test)
        test_error = mean_squared_error(y_test, y_test_pred)

        return train_error, test_error

    def predict(self, X):
        return self.model.predict(X)
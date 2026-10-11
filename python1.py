
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Sample dataset: Area in square feet, price in lakhs
X = np.array([
    [500], [700], [800], [1000], [1200],
    [1500], [1800], [2000], [2200], [2500]
])

y = np.array([20, 28, 32, 40, 48, 60, 72, 80, 88, 100])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel:", name)
    print("Actual Prices:", y_test)
    print("Predicted Prices:", np.round(y_pred, 2))
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))

# Predict price for a new house
model = LinearRegression()
model.fit(X_train, y_train)

area = np.array([[1600]])
price = model.predict(area)

print("\nArea:", area[0][0], "sq. ft.")
print("Predicted Price:", round(price[0], 2), "lakhs")

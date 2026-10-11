
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Sample dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# Initialize parameters
m = 0.0
c = 0.0

learning_rate = 0.01
epochs = 1000
n = len(X)

loss_history = []

# Apply Gradient Descent
for i in range(epochs):
    y_pred = m * X + c

    # Calculate gradients
    dm = (-2 / n) * np.sum(X * (y - y_pred))
    dc = (-2 / n) * np.sum(y - y_pred)

    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    # Calculate and store loss
    predictions = m * X + c
    loss = mean_squared_error(y, predictions)
    loss_history.append(loss)

# Display results
print("Slope:", round(m, 4))
print("Intercept:", round(c, 4))
print("Final MSE:", round(loss_history[-1], 6))
print("Predictions:", np.round(m * X + c, 2))

# Plot loss graph
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.show()

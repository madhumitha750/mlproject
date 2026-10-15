import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
# Features: study hours, attendance, previous marks
X = np.array([
    [1, 50, 40],
    [2, 55, 45],
    [3, 60, 50],
    [4, 65, 55],
    [5, 70, 60],
    [6, 75, 65],
    [7, 80, 70],
    [8, 85, 75],
    [9, 90, 85],
    [10, 95, 90]
])

# Target: expected marks
y = np.array([35, 40, 45, 50, 55, 65, 70, 78, 88, 95])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("ML model trained and saved successfully!")

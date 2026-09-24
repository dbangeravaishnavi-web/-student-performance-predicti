# Student Performance Prediction Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create sample student data
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [60, 65, 70, 72, 75, 80, 82, 85, 90, 95],
    "assignment_score": [45, 50, 55, 60, 65, 70, 75, 80, 88, 92],
    "final_score": [42, 48, 55, 58, 65, 70, 74, 80, 87, 94]
}

df = pd.DataFrame(data)

# Input features and target
X = df[["study_hours", "attendance", "assignment_score"]]
y = df["final_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Student Performance Prediction")
print("--------------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Predict performance for a new student
new_student = [[6, 85, 78]]
predicted_score = model.predict(new_student)

print("Predicted Final Score:", round(predicted_score[0], 2))
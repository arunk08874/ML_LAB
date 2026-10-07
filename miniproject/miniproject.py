import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Display dataset
print("Dataset:")
print(data.head())

# Create target variable
# Example: crops are treated as suitable classes
le = LabelEncoder()
data["label"] = le.fit_transform(data["label"])

# Input features
X = data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]

# Target
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nWeather-Based Crop Suitability Prediction")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Example prediction
sample = [[90, 42, 43, 25.5, 80.0, 6.5, 200]]

prediction = model.predict(sample)

crop = le.inverse_transform(prediction)

print("Predicted Suitable Crop:", crop[0])
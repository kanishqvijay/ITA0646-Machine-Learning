
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
data = pd.read_csv('iris.csv')
X = data.drop('species', axis=1).values
y = data['species'].values

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Logistic Regression model
lr = LogisticRegression(random_state=42, multi_class='ovr')
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("
Classification Report:")
print(classification_report(y_test, y_pred))

# Predict a new sample
new_sample = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
prediction = lr.predict(new_sample)
print(f"Predicted class for new sample: {prediction[0]}")
        

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

# Load data
data = pd.read_csv('iris.csv')

# Preprocess data
X = data.drop('species', axis=1)
y = data['species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# Make predictions
y_pred = nb.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("
Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a new sample
new_sample = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
prediction = nb.predict(new_sample)
print(f"
Predicted class for new sample: {prediction[0]}")
        
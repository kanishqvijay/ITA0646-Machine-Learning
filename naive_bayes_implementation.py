
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = []
        for c in self.classes:
            X_c = X[y == c]
            self.parameters.append({
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0),
                'prior': len(X_c) / len(X)
            })

    def _pdf(self, x, mean, var):
        return np.exp(-(x-mean)**2 / (2*var)) / np.sqrt(2*np.pi*var)

    def _predict(self, x):
        posteriors = []
        for params in self.parameters:
            prior = np.log(params['prior'])
            posterior = np.sum(np.log(self._pdf(x, params['mean'], params['var'])))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

# Load data
data = pd.read_csv('iris.csv')

# Preprocess data
X = data.drop('species', axis=1).values
y = data['species'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
nb = NaiveBayes()
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
        
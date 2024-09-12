
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_ih = np.random.randn(self.input_size, self.hidden_size)
        self.weights_ho = np.random.randn(self.hidden_size, self.output_size)
        
        self.bias_h = np.zeros((1, self.hidden_size))
        self.bias_o = np.zeros((1, self.output_size))
        
    def forward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights_ih) + self.bias_h)
        self.output = sigmoid(np.dot(self.hidden, self.weights_ho) + self.bias_o)
        return self.output
        
    def backward(self, X, y, output, learning_rate):
        error = y - output
        d_output = error * sigmoid_derivative(output)
        
        error_hidden = np.dot(d_output, self.weights_ho.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden)
        
        self.weights_ho += learning_rate * np.dot(self.hidden.T, d_output)
        self.bias_o += learning_rate * np.sum(d_output, axis=0, keepdims=True)
        
        self.weights_ih += learning_rate * np.dot(X.T, d_hidden)
        self.bias_h += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)
    
    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
    
    def predict(self, X):
        return self.forward(X)

# Load and preprocess data
data = pd.read_csv('iris.csv')
X = data.drop('species', axis=1).values
y = pd.get_dummies(data['species']).values

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the neural network
nn = NeuralNetwork(input_size=4, hidden_size=5, output_size=3)
nn.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# Make predictions
predictions = nn.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(true_classes, predicted_classes)
print(f"Accuracy: {accuracy}")
        
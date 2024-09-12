
import pandas as pd
import numpy as np

def find_s(data):
    # Initialize hypothesis with the first positive example
    h = list(data[data['EnjoySport'] == 'Yes'].iloc[0])
    
    # Iterate through all positive examples
    for _, row in data[data['EnjoySport'] == 'Yes'].iterrows():
        for i in range(len(h)):
            if row[i] != h[i]:
                h[i] = '?'
    
    return h

# Read data from CSV file
data = pd.read_csv('enjoysport.csv')

# Apply FIND-S algorithm
result = find_s(data)

print("Most specific hypothesis:")
print(result)
        
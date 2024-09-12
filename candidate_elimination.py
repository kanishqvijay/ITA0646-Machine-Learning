
import pandas as pd
import numpy as np

def candidate_elimination(data):
    attributes = list(data.columns)[:-1]
    num_attributes = len(attributes)
    
    # Initialize G and S
    G = [['?' for i in range(num_attributes)]]
    S = ['0' for i in range(num_attributes)]
    
    # Iterate through examples
    for index, row in data.iterrows():
        if row['EnjoySport'] == 'Yes':
            # Positive example
            for i in range(len(S)):
                if S[i] == '0':
                    S[i] = row[i]
                elif S[i] != row[i]:
                    S[i] = '?'
            
            G = [g for g in G if all(g[i] == '?' or g[i] == row[i] for i in range(num_attributes))]
        
        else:
            # Negative example
            G = [g for g in G if any(g[i] != '?' and g[i] != row[i] for i in range(num_attributes))]
            
            new_G = []
            for g in G:
                for i in range(num_attributes):
                    if g[i] == '?' or g[i] == row[i]:
                        g_new = g.copy()
                        g_new[i] = '0' if row[i] == '1' else '1'
                        new_G.append(g_new)
            G.extend(new_G)
    
    return S, G

# Read data from CSV file
data = pd.read_csv('enjoysport.csv')

# Apply Candidate-Elimination algorithm
S, G = candidate_elimination(data)

print("Most specific hypothesis S:")
print(S)
print("
Most general hypotheses G:")
for g in G:
    print(g)
        
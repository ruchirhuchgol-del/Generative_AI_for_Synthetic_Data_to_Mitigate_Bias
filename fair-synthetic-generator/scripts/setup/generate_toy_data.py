import pandas as pd
import numpy as np

# Create a tiny dummy dataset for the 'adult' schema
data = {
    'age': [25, 45, 30, 50, 22],
    'workclass': ['Private', 'Self-emp-not-inc', 'Private', 'Private', 'Local-gov'],
    'fnlwgt': [226802, 89814, 33114, 15822, 1222],
    'education': ['11th', 'HS-grad', 'Some-college', '7th-8th', 'Preschool'],
    'educational-num': [7, 9, 10, 4, 1],
    'marital-status': ['Never-married', 'Married-civ-spouse', 'Never-married', 'Divorced', 'Separated'],
    'occupation': ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv', 'Other-service', 'Priv-house-serv'],
    'relationship': ['Own-child', 'Husband', 'Unmarried', 'Unmarried', 'Not-in-family'],
    'race': ['Black', 'White', 'Black', 'White', 'Black'],
    'gender': ['Male', 'Male', 'Female', 'Male', 'Female'],
    'capital-gain': [0, 0, 0, 0, 0],
    'capital-loss': [0, 0, 0, 0, 0],
    'hours-per-week': [40, 50, 40, 30, 20],
    'native-country': ['United-States', 'United-States', 'United-States', 'United-States', 'United-States'],
    'income': ['<=50K', '>50K', '<=50K', '<=50K', '<=50K']
}

df = pd.DataFrame(data)
df.to_csv('data/raw/toy_adult.csv', index=False)
print("Created data/raw/toy_adult.csv")

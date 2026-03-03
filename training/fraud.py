import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle
import warnings

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")

print("Libraries loaded successfully")

# Load dataset
data = pd.read_csv("data/PS_20174392719_1491204439457_log.csv").sample(200000)

print(data.head())


# Show last rows
print("\nLast 5 rows:")
print(data.tail())

# Dataset information
print("\nDataset Info:")
print(data.info())

# Check null values
print("\nNull values in dataset:")
print(data.isnull().sum())

# Correlation matrix
print("\nCorrelation matrix:")
print(data.corr(numeric_only=True))

# Heatmap visualization
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Check fraud vs normal transactions
print("\nFraud vs Normal Transactions:")
print(data['isFraud'].value_counts())

# Convert transaction type to numbers
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

# Prepare input and output
X = data.drop(['nameOrig','nameDest','isFraud','isFlaggedFraud'], axis=1)
y = data['isFraud']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
print("\nModel Accuracy:", accuracy_score(y_test, pred))

# Save trained model
pickle.dump(model, open("payments.pkl", "wb"))
print("\nModel saved as payments.pkl")
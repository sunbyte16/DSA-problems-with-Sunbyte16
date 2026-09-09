import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create dataset
df = pd.DataFrame({
    'area': [500, 600, 700, 800, 900, 1000],
    'prices': [20, 25, 30, 35, 40, 45]
})

# Display data
print("Shape:", df.shape)

print("\nHead of DataFrame:")
print(df.head())

print("\nColumns:")
print(df.columns)

# Input and output
X = df['area']
y = df['prices']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=104,
    shuffle=True
)

# Display results
print("\nX_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)

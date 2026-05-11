import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('ml/data/creditcard.csv')
y = df['Class'].values

# Same split as preprocess.py
_, X_temp, _, y_temp = train_test_split(df, y, test_size=0.30, random_state=42, stratify=y)
_, X_test, _, _ = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# All fraud (74) + 500 random normal from test set
fraud = X_test[X_test['Class'] == 1]
normal = X_test[X_test['Class'] == 0].sample(n=500, random_state=42)
sample = pd.concat([fraud, normal]).sample(frac=1, random_state=42)

sample.to_csv('ml/data/test_sample.csv', index=False)
print(f"Saved: {len(sample)} rows (fraud: {len(fraud)}, normal: {len(normal)})")
import pandas as pd

# Load data
df = pd.read_csv("data/titanic.csv")

# Select columns
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Encode categorical
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# Save processed data
df.to_csv("processed/processed.csv", index=False)

print("Preprocessing completed.")
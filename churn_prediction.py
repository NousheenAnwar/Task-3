# ============================================================
#  Customer Churn Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

df = pd.read_csv("Churn_Modelling.csv")

print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# Drop columns that are NOT useful for prediction
# RowNumber, CustomerId, Surname are just identifiers — not real features
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

print("\nAfter dropping useless columns:")
print(df.head())

# --- 3a. Label Encoding for 'Gender' (Male=1, Female=0) ---
# Good when there are only 2 categories
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
print("\nGender after Label Encoding (0=Female, 1=Male):")
print(df["Gender"].value_counts())

# --- 3b. One-Hot Encoding for 'Geography' ---
# Good when there are 3+ categories (France, Spain, Germany)
# drop_first=True avoids the "dummy variable trap"
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

print("\nColumns after One-Hot Encoding Geography:")
print(df.columns.tolist())

# X = all columns EXCEPT 'Exited'
# y = 'Exited' column (1 = customer left, 0 = customer stayed)

X = df.drop(columns=["Exited"])
y = df["Exited"]

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)
print("\nChurn distribution:\n", y.value_counts())

# Models work better when all numbers are on a similar scale
# StandardScaler makes the mean=0 and std=1

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 80% of data for training, 20% for testing
# random_state=42 makes results reproducible

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")


# Random Forest = a group of Decision Trees working together
# n_estimators=100 means we build 100 trees

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n✅ Model training complete!")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))


fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Customer Churn Prediction — Results", fontsize=15, fontweight="bold")

# --- Plot 1: Churn Distribution ---
churn_counts = y.value_counts()
axes[0].bar(["Stayed (0)", "Churned (1)"], churn_counts.values,
            color=["steelblue", "salmon"], edgecolor="black")
axes[0].set_title("Churn Distribution in Dataset")
axes[0].set_ylabel("Number of Customers")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

# --- Plot 2: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Stayed", "Churned"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix")

# --- Plot 3: Feature Importance ---
feature_names = X.columns.tolist()
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

axes[2].barh(
    [feature_names[i] for i in indices],
    importances[indices],
    color="steelblue", edgecolor="black"
)
axes[2].invert_yaxis()
axes[2].set_title("Feature Importance (Random Forest)")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("churn_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📈 Plot saved as 'churn_results.png'")


print("\n" + "="*55)
print("🔍 TOP 5 FEATURES THAT INFLUENCE CHURN:")
print("="*55)
for rank, i in enumerate(indices[:5], 1):
    print(f"  {rank}. {feature_names[i]:<25} → {importances[i]:.4f}")

print("""
📌 WHAT DO THESE RESULTS MEAN?
-----------------------------------------------
• Accuracy ~86%: The model correctly predicts whether
  a customer will churn 86% of the time.

• Feature Importance tells us WHICH factors matter most
  for predicting churn. Higher score = more influence.

• Common top features in this dataset:
  - Age          → Older customers churn more
  - NumOfProducts→ Having 1 or 3+ products increases risk
  - IsActiveMember→ Inactive members are more likely to leave
  - Balance      → High balance but no activity = churn risk
  - Geography    → Germany has higher churn rate

• Confusion Matrix:
  - True Positives (bottom-right): Correctly predicted churners
  - False Negatives (bottom-left): Missed churners (most costly!)
""")

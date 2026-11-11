
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Define features
FEATURES = [
    "avg_usage",
    "peak_usage",
    "off_peak_usage",
    "variance"
]
TARGET = "theft"

# 2. Load dataset
df = pd.read_csv("smart_meter_data.csv")
print("Dataset columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# 3. Validate dataset columns
required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

df = df[required_columns].copy()

# 4. Convert columns to numeric
for col in required_columns:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

df = df.dropna()

# 5. Separate features and target
X = df[FEATURES]
y = df[TARGET]

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "Target column must contain both 0 and 1."
    )

if y.value_counts().min() < 2:
    raise ValueError(
        "At least two samples per class are required."
    )

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# 8. Train model
model.fit(X_train, y_train)

# 9. Make predictions
y_pred = model.predict(X_test)

# 10. Evaluate model
metrics = {
    "accuracy": float(
        accuracy_score(y_test, y_pred)
    ),
    "precision": float(
        precision_score(
            y_test, y_pred, pos_label=1,
            zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_test, y_pred, pos_label=1,
            zero_division=0
        )
    ),
    "f1_score": float(
        f1_score(
            y_test, y_pred, pos_label=1,
            zero_division=0
        )
    )
}

print("\n===== MODEL EVALUATION =====")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# 11. Save metrics
with open("model_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# 12. Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Theft"],
        zero_division=0
    )
)

# 13. Generate confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Theft"]
)

display.plot(cmap="Blues")

plt.title("Energy Theft Detection")
plt.savefig(
    "confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

# 14. Save trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel training completed!")
print("Model saved as model.pkl")
print("Metrics saved as model_metrics.json")
print("Confusion matrix saved as confusion_matrix.png")

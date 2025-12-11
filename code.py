import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1. Load Dataset
# ---------------------------------------------
df = pd.read_csv("/Workspace/Users/gulamdustgeer124@gmail.com/lungz cancer.csv")

# (Optional) standardize column names: remove spaces, uppercase, underscores
df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
print("Columns:", df.columns.tolist())

# ---------------------------------------------
# 2. Clean Data
# ---------------------------------------------
df.drop_duplicates(inplace=True)
df = df.fillna(df.mode().iloc[0])

# ---------------------------------------------
# 3. Encode Categorical Columns
# ---------------------------------------------
gender_le = LabelEncoder()
target_le = LabelEncoder()

# Encode GENDER (M/F)
df["GENDER"] = gender_le.fit_transform(df["GENDER"])

# Encode LUNG_CANCER (YES/NO)
df["LUNG_CANCER"] = target_le.fit_transform(df["LUNG_CANCER"])
print("Target mapping:", dict(zip(target_le.classes_, target_le.transform(target_le.classes_))))

# ---------------------------------------------
# 4. Split Features & Target
# ---------------------------------------------
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# ---------------------------------------------
# 5. Train-Test Split
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------
# 6. Train Random Forest Classifier
# ---------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced"   # handles any class imbalance
)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# ---------------------------------------------
# 7. Evaluation
# ---------------------------------------------
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------------------------
# 8. Feature Importance Plot
# ---------------------------------------------
importances = rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,6))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 9. Predict a Single Patient (by index from dataset)
# ---------------------------------------------
idx = 16  # choose any row you want to inspect

new_patient_raw = X.iloc[[idx]]             # keep as DataFrame
true_label = target_le.inverse_transform([y.iloc[idx]])[0]
pred_encoded = rf_model.predict(new_patient_raw)[0]
pred_label = target_le.inverse_transform([pred_encoded])[0]

print("\n--- Prediction for Patient at Index", idx, "---")
print("Features:\n", new_patient_raw)
print("True label from dataset   :", true_label)
print("Model predicted label     :", pred_label)
print("Result (your wording)     :", 
      "Malignant (Cancerous)" if pred_encoded == 1 else "Benign (Not Cancerous)")



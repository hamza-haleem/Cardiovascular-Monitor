# ============================================
#  HEART DISEASE MODEL TRAINING SCRIPT
# ============================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv("heart.csv")   # <-- change filename if needed

target_col = "HeartDisease"
y = df[target_col]
X = df.drop(columns=[target_col])

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# ============================================
# PREPROCESSOR
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

joblib.dump(preprocessor, "preprocessor.pkl")
print("Saved preprocessor: preprocessor.pkl")

# ============================================
# MODELS TO TRAIN
# ============================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4
    )
}

# ============================================
# TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# TRAIN + EVALUATE ALL MODELS
# ============================================

results = []

best_model_name = None
best_accuracy = -1
best_model = None

print("\nTraining models...\n")

for name, model in models.items():

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append([name, acc, prec, rec, f1])

    print(f"{name}: ACC={acc:.3f}  PREC={prec:.3f}  REC={rec:.3f}  F1={f1:.3f}")

    # track best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model = clf

print("\n====================================================")
print(f"BEST MODEL: {best_model_name}  (Accuracy: {best_accuracy:.3f})")
print("====================================================\n")

# ============================================
# SAVE COMPARISON RESULTS
# ============================================

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

results_df.to_csv("model_performance.csv", index=False)
print("Saved model comparison: model_performance.csv")

# ============================================
# SAVE BEST MODEL
# ============================================

joblib.dump(best_model, "best_heart_model.pkl")
print("Saved best model: best_heart_model.pkl")

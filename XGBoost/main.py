import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------
# LOAD DATA
# -------------------------
data = pd.read_csv("training_data_v2.csv")

print("Dataset shape:", data.shape)

# -------------------------
# ENCODE FEATURE + VALUE
# -------------------------
data["feature_value"] = data["feature"] + "_" + data["value"]

data = pd.get_dummies(data, columns=["feature_value"])

data = data.drop(["feature", "value"], axis=1)

# -------------------------
# SPLIT FEATURES / TARGET
# -------------------------
X = data.drop("target_score", axis=1)
y = data["target_score"]

print("Feature shape:", X.shape)

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MODEL
# -------------------------
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# -------------------------
# TRAIN
# -------------------------
print("\nTraining model...")
model.fit(X_train, y_train)

# -------------------------
# SAVE MODEL + COLUMNS
# -------------------------
joblib.dump(model, "xgb_model.pkl")
joblib.dump(X.columns, "model_columns.pkl")

print("\nModel saved!")

# -------------------------
# PREDICT
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# EVALUATE
# -------------------------
mse = mean_squared_error(y_test, y_pred)
print("\nMSE:", mse)

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 15 Important Features:")
print(importance_df.head(15))
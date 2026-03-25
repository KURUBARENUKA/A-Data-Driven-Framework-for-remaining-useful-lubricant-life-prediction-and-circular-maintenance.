import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# ------------------------------------------
# 1) Load the mapped dataset
# ------------------------------------------
df = pd.read_csv("kaggle_dataset.csv")  # change filename if needed

# ------------------------------------------
# 2) Select input features (X) and target (y)
# ------------------------------------------
features = [
    "temp", "vibration", "pressure", "load",
    "runtime_hours", "contamination",
    "oxidation_index", "viscosity_drift",
    "thermal_stress", "mechanical_stress",
    "contamination_prob"
]

target = "rull"   # remaining useful lubricant life

X = df[features]
y = df[target]

# ------------------------------------------
# 3) Train / Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# 4) Train the model
# ------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
model.fit(X_train, y_train)

# ------------------------------------------
# 5) Evaluate model performance
# ------------------------------------------
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = mse ** 0.5
r2 = r2_score(y_test, preds)

print("\n========== MODEL TRAINED SUCCESSFULLY ==========")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print("===============================================\n")

# ------------------------------------------
# 6) Save the model
# ------------------------------------------
joblib.dump(model, "ecotwin_rull_model.joblib")
print("Model saved as: ecotwin_rull_model.joblib\n")

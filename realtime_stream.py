import pandas as pd
import joblib
import time
import os
import random

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = "kaggle_dataset.csv"          # mapped dataset
MODEL_FILE = "ecotwin_rull_model.joblib"  # trained model
DELAY_SECONDS = 0.5                       # pause between "live" readings

# Thresholds for alerts (tune these)
RULL_WARN = 300       # below this → warning
RULL_CRITICAL = 100   # below this → critical

# -----------------------------
# 1) Load data & model
# -----------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# Make sure required columns exist
FEATURES = [
    "temp", "vibration", "pressure", "load",
    "runtime_hours", "contamination",
    "oxidation_index", "viscosity_drift",
    "thermal_stress", "mechanical_stress",
    "contamination_prob"
]

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"These required feature columns are missing in the dataset: {missing}")

print("Loading model...")
model = joblib.load(MODEL_FILE)

print("\n=== EcoTwin Real-Time Monitoring Started ===")
print("Press Ctrl + C to stop.\n")

# -----------------------------
# 2) Helper to classify health status
# -----------------------------
def classify_status(pred_rull, contamination_prob):
    """
    Simple rule-based status from RULL and contamination.
    """
    if pred_rull <= RULL_CRITICAL:
        level = "CRITICAL"
        msg = "Immediate regeneration required!"
    elif pred_rull <= RULL_WARN:
        level = "WARNING"
        msg = "Plan regeneration soon."
    else:
        level = "OK"
        msg = "Lubricant condition acceptable."

    # Add note for high contamination
    if contamination_prob > 0.7:
        level = "CRITICAL"
        msg = "High contamination! Regeneration + filtration required."

    return level, msg

# -----------------------------
# 3) Streaming loop
# -----------------------------

# For realism, shuffle the order (optional)
indices = list(df.index)
random.shuffle(indices)

try:
    for i, idx in enumerate(indices):
        row = df.loc[idx]

        # Build model input
        x = row[FEATURES].to_frame().T  # 1-row DataFrame

        # Predict RULL
        pred_rull = model.predict(x)[0]

        # Determine status
        level, msg = classify_status(pred_rull, row["contamination_prob"])

        # Pretty print
        print(f"Cycle #{i+1} (row {idx})")
        print(f"  Temp: {row['temp']:.2f} °C | Vibration: {row['vibration']:.3f} mm/s | Pressure: {row['pressure']:.2f} bar")
        print(f"  Predicted RULL: {pred_rull:.2f}")
        print(f"  Status: {level} → {msg}")
        print("-" * 60)

        time.sleep(DELAY_SECONDS)

    print("\n=== Simulation finished (end of dataset) ===")

except KeyboardInterrupt:
    print("\n=== Monitoring stopped by user (Ctrl + C) ===")

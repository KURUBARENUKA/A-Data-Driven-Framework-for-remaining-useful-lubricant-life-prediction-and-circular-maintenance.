# ============================================================
#   ECO-TWIN DIGITAL TWIN MAPPING CODE
#   Condition Monitoring of Hydraulic Systems Dataset
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# STEP 1 — LOAD RAW SENSOR FILES
# ============================================================

# PRESSURE SENSORS (100 Hz)
PS1 = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\PS1.txt", 
                  sep="\t", header=None).apply(pd.to_numeric, errors='coerce')
PS2 = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\PS2.txt", 
                  sep="\t", header=None).apply(pd.to_numeric, errors='coerce')
PS3 = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\PS3.txt", 
                  sep="\t", header=None).apply(pd.to_numeric, errors='coerce')
PS4 = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\PS4.txt", 
                  sep="\t", header=None).apply(pd.to_numeric, errors='coerce')
PS5 = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\PS5.txt", 
                  sep="\t", header=None).apply(pd.to_numeric, errors='coerce')
PS6 = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\PS6.txt", 
                  sep="\t", header=None).apply(pd.to_numeric, errors='coerce')

# TEMPERATURE SENSORS (1 Hz)
TS1 = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\TS1.txt", 
                  sep="\t", header=None).apply(pd.to_numeric, errors='coerce')
TS2 = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\TS2.txt", 
                  sep="\t", header=None).apply(pd.to_numeric, errors='coerce')
TS3 = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\TS3.txt", 
                  sep="\t", header=None).apply(pd.to_numeric, errors='coerce')
TS4 = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\TS4.txt", 
                  sep="\t", header=None).apply(pd.to_numeric, errors='coerce')

# VIBRATION SENSOR (1 Hz)
VS1 = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\VS1.txt", 
                  sep="\t", header=None).apply(pd.to_numeric, errors='coerce')

# PROFILE FILE (pump leakage, accumulator pressure, etc.)
profile = pd.read_csv(r"C:\Users\navee\Downloads\CIRCULAR_AI_ECONOMY\Dataset\Condition Monitoring of Hydraulic Systems(Kaggle)\profile.txt", 
                      sep="\t", header=None)

# Ensure profile has exactly 5 columns
if profile.shape[1] == 5:
    profile.columns = ["cooler", "valve", "pump_leakage", "acc_pressure", "motor_power"]
else:
    raise ValueError(f"Profile file has {profile.shape[1]} columns; expected 5.")

# ============================================================
# STEP 2 — AGGREGATE HIGH-FREQUENCY SENSOR DATA
# ============================================================

# Mean of each row (cycle) for pressure sensors
pressure = pd.DataFrame({
    "PS1": PS1.mean(axis=1),
    "PS2": PS2.mean(axis=1),
    "PS3": PS3.mean(axis=1),
    "PS4": PS4.mean(axis=1),
    "PS5": PS5.mean(axis=1),
    "PS6": PS6.mean(axis=1)
})

# Mean of each row for temperature sensors
temperature = pd.DataFrame({
    "TS1": TS1.mean(axis=1),
    "TS2": TS2.mean(axis=1),
    "TS3": TS3.mean(axis=1),
    "TS4": TS4.mean(axis=1)
})

# Vibration (row-wise mean)
vibration = VS1.mean(axis=1)

# ============================================================
# STEP 3 — BUILD DIGITAL TWIN BASE DATAFRAME
# ============================================================

df = pd.DataFrame()

# Temperature (average of TS sensors)
df["temp"] = temperature.mean(axis=1)

# Vibration
df["vibration"] = vibration

# Pressure (average of all PS sensors)
df["pressure"] = pressure.mean(axis=1)

# Load (normalized pressure)
df["load"] = df["pressure"] / df["pressure"].max()

# Runtime (cycle_index / 60 → hours)
df["runtime_hours"] = df.index / 60

# Contamination level (pump leakage normalized 0–1)
df["contamination"] = profile["pump_leakage"] / 2

# ============================================================
# STEP 4 — LUBRICANT LIFECYCLE DERIVED VARIABLES
# ============================================================

# Oxidation Index
df["oxidation_index"] = ((df["temp"] - 60) / 10).clip(lower=0)

# Viscosity Drift
df["viscosity_drift"] = 0.5*(df["runtime_hours"]/1000) + 0.5*(df["temp"]/100)

# Thermal Stress
df["thermal_stress"] = (df["temp"]**2) / 10000

# Mechanical Stress
def mech_stress(v):
    if v < 2:
        return 0
    elif v < 4.5:
        return 1
    else:
        return 2

df["mechanical_stress"] = df["vibration"].apply(mech_stress)

# Contamination Probability
df["contamination_prob"] = df["contamination"]

# ============================================================
# STEP 5 — SYNTHETIC REMAINING USEFUL LUBRICANT LIFE (RULL)
# ============================================================

max_rul = len(df)
df["rull"] = max_rul - df.index

# ============================================================
# STEP 6 — SAVE FINAL DIGITAL TWIN DATASET
# ============================================================

df.to_csv("kaggle_dataset.csv", index=False)
print("\n=========================================")
print("EcoTwin Digital Twin Dataset Created!")
print("Saved as: kaggle_dataset.csv")
print("Rows:", df.shape[0], "Columns:", df.shape[1])
print("=========================================\n")

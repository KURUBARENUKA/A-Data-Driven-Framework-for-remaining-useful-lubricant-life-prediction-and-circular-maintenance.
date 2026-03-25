import joblib
import pandas as pd

model = joblib.load("ecotwin_rull_model.joblib")

sample = {
    "temp": 72.3,
    "vibration": 4.1,
    "pressure": 145,
    "load": 0.63,
    "runtime_hours": 160.2,
    "contamination": 0.32,
    "oxidation_index": 0.65,
    "viscosity_drift": 0.40,
    "thermal_stress": 0.52,
    "mechanical_stress": 1,
    "contamination_prob": 0.32
}

df = pd.DataFrame([sample])
prediction = model.predict(df)[0]
print("Predicted RULL (Remaining Useful Lubricant Life):", prediction)

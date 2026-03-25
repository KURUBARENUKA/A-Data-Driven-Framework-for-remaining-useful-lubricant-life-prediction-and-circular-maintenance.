import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =========================================
# PAGE CONFIG — MUST BE FIRST STREAMLIT CMD
# =========================================
st.set_page_config(
    page_title="EcoTwin – RULL Digital Twin Dashboard",
    layout="wide"
)

# ==============================
# CONFIG
# ==============================
DATA_FILE = "kaggle_dataset.csv"
MODEL_FILE = "ecotwin_rull_model.joblib"

FEATURES = [
    "temp", "vibration", "pressure", "load",
    "runtime_hours", "contamination",
    "oxidation_index", "viscosity_drift",
    "thermal_stress", "mechanical_stress",
    "contamination_prob"
]

TARGET = "rull"

# Thresholds for status
RULL_WARN = 300
RULL_CRITICAL = 100

# ==============================
# LOAD DATA & MODEL
# ==============================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# --- load dataset ---
try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error(f"Could not load dataset '{DATA_FILE}': {e}")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"These required columns are missing in the dataset: {missing}")
    st.stop()

# --- load model ---
try:
    model = load_model(MODEL_FILE)
except Exception as e:
    st.error(f"Could not load model '{MODEL_FILE}': {e}")
    st.stop()

# --- init session state ---
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "live_index" not in st.session_state:
    st.session_state.live_index = 0

if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ==============================
# HELPER: STATUS & ALERTS
# ==============================
def classify_status(pred_rull, contamination_prob):
    if pred_rull <= RULL_CRITICAL:
        level = "CRITICAL"
        msg = "🚨 Immediate regeneration required!"
        color = "red"
    elif pred_rull <= RULL_WARN:
        level = "WARNING"
        msg = "⚠️ Plan regeneration soon."
        color = "orange"
    else:
        level = "OK"
        msg = "✅ Lubricant condition acceptable."
        color = "green"

    if contamination_prob > 0.7:
        level = "CRITICAL"
        msg = "🚨 High contamination! Regeneration + filtration required."
        color = "red"

    return level, msg, color

def create_alert_text(idx, pred_rull, level, msg):
    return (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"CYCLE {idx} | STATUS: {level} | RULL={pred_rull:.2f} | {msg}"
    )

# ==============================
# RETRAIN FUNCTION (MODEL AGENT)
# ==============================
def retrain_model(df, features, target_col=TARGET, model_path=MODEL_FILE):
    from sklearn.ensemble import RandomForestRegressor

    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    new_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )
    new_model.fit(X_train, y_train)

    preds = new_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    joblib.dump(new_model, model_path)
    return new_model, rmse, r2

# ==============================
# PDF REPORT GENERATION (REPORTING AGENT)
# ==============================
def generate_pdf_report(df):
    avg_rull = df["rull"].mean()
    min_rull = df["rull"].min()
    max_rull = df["rull"].max()
    avg_temp = df["temp"].mean()
    avg_vib = df["vibration"].mean()
    avg_contam = df["contamination_prob"].mean()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "EcoTwin - Lubricant Health Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Summary Metrics", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Average RULL: {avg_rull:.2f}", ln=True)
    pdf.cell(0, 8, f"Min RULL: {min_rull:.2f}", ln=True)
    pdf.cell(0, 8, f"Max RULL: {max_rull:.2f}", ln=True)
    pdf.cell(0, 8, f"Average Temperature: {avg_temp:.2f} °C", ln=True)
    pdf.cell(0, 8, f"Average Vibration: {avg_vib:.3f} mm/s", ln=True)
    pdf.cell(0, 8, f"Average Contamination Probability: {avg_contam:.2f}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Alerts Summary", ln=True)
    pdf.set_font("Arial", "", 12)

    if st.session_state.alerts:
        for a in st.session_state.alerts[-10:]:
            pdf.multi_cell(0, 6, f"- {a}")
    else:
        pdf.cell(0, 8, "No critical alerts logged in this session.", ln=True)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# ==============================
# MODEL METRICS (for Assistant & Explainability)
# ==============================
@st.cache_data
def compute_model_metrics(df, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)
    return {"rmse": rmse, "r2": r2}

metrics_info = compute_model_metrics(df, FEATURES, TARGET)

# Feature importance for assistant & explain tab
feature_importance_df = None
if hasattr(model, "feature_importances_"):
    feature_importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

# ==============================
# ASSISTANT AGENT: QA LOGIC
# ==============================
def answer_user_query(question: str) -> str:
    q = question.lower()

    # Project overview
    if "what is this project" in q or ("project" in q and "do" in q):
        return (
            "This project is a **software-only digital twin** for lubricant lifecycle monitoring. "
            "It uses the 'Condition Monitoring of Hydraulic Systems' dataset from Kaggle, maps raw "
            "sensor signals (pressure, temperature, vibration, leakage) into lubrication indicators "
            "like oxidation index, viscosity drift, thermal stress, contamination probability and "
            "Remaining Useful Lubricant Life (RULL), and then uses a RandomForest model to predict RULL. "
            "The Streamlit dashboard shows live monitoring, what-if simulations, alerts, and a PDF report."
        )

    # RULL explanation
    if "rull" in q or "remaining useful" in q:
        return (
            "RULL stands for **Remaining Useful Lubricant Life**. It is a regression target that estimates "
            "how much usable life is left in the lubricant before it should be regenerated or replaced. "
            "In this project, RULL is learned from patterns in temperature, vibration, pressure, runtime "
            "and contamination-related indicators."
        )

    # Data / dataset questions
    if "data" in q or "dataset" in q:
        return (
            f"The system uses a mapped version of the **Condition Monitoring of Hydraulic Systems** dataset. "
            f"It contains {df.shape[0]} cycles and {df.shape[1]} columns. Each row represents one operating cycle "
            f"of the hydraulic system with aggregated sensor values. From these, we derive lubrication indicators "
            f"like oxidation_index, viscosity_drift, thermal_stress, mechanical_stress, contamination_prob and rull."
        )

    # Features / most important features
    if "feature" in q or "important" in q or "importance" in q:
        if feature_importance_df is not None:
            top_feats = feature_importance_df.head(5)
            lst = ", ".join(f"{r['feature']}" for _, r in top_feats.iterrows())
            return (
                "The model uses engineered features such as temperature, vibration, pressure, load, runtime, "
                "contamination, oxidation_index, viscosity_drift, thermal_stress, mechanical_stress and "
                "contamination_prob.\n\n"
                f"According to the RandomForest feature importances, the most influential features (top 5) are: {lst}."
            )
        else:
            return (
                "The model relies on engineered features including temperature, vibration, pressure, load, runtime, "
                "contamination, oxidation_index, viscosity_drift, thermal_stress, mechanical_stress and "
                "contamination_prob. This build does not expose tree-based feature_importances_, but those are the "
                "core drivers of RULL prediction."
            )

    # Metrics / accuracy / RMSE / R2
    if "rmse" in q or "accuracy" in q or "score" in q or "performance" in q or "r2" in q:
        return (
            f"The current RandomForest RULL model was evaluated using a held-out test split. "
            f"It achieves an RMSE of approximately **{metrics_info['rmse']:.3f}** and an R² score of "
            f"approximately **{metrics_info['r2']:.3f}**. These values indicate how well the model can predict "
            f"remaining lubricant life from the input indicators."
        )

    # Contamination / alerts / critical state
    if "alert" in q or "critical" in q or "warning" in q or "contamination" in q:
        return (
            "The live monitoring tab classifies each prediction into **OK**, **WARNING** or **CRITICAL**. "
            "The classification depends mainly on predicted RULL and contamination probability. "
            "If RULL is below certain thresholds (e.g., < 100 cycles) or contamination probability is very high, "
            "the system marks the state as CRITICAL and logs a time-stamped alert which can be viewed in the sidebar "
            "and included in the generated PDF report."
        )

    # Why Kaggle / why no hardware
    if "kaggle" in q or "real factory" in q or "hardware" in q:
        return (
            "Real lubricant lifecycle and sensor data from factories is typically proprietary and not publicly shared. "
            "The 'Condition Monitoring of Hydraulic Systems' dataset is a well-known public benchmark that contains "
            "real sensor signals for pressure, temperature, vibration and leakage. By mapping these signals into "
            "lubrication-related indicators, we can prototype and validate a digital twin and ML pipeline **without "
            "needing physical hardware or direct factory integration**. Later, the same pipeline can be connected to "
            "live sensors."
        )

    # Smart agents architecture explanation
    if "agent" in q or "agents" in q or "smart" in q:
        return (
            "The system follows a smart-agent style architecture:\n"
            "• **Data Agent** – implemented through the preprocessing/mapping script that converts raw hydraulic sensor data\n"
            "  into lubricant lifecycle indicators like oxidation_index, viscosity_drift, thermal_stress and RULL.\n"
            "• **Model Agent** – implemented through the training code and the 'Retrain Model (Auto-Agent)' button which retrains\n"
            "  the RandomForest model when needed.\n"
            "• **Monitoring Agent** – implemented in the Live Monitor tab, which simulates cycles, predicts RULL and logs alerts\n"
            "  when conditions become critical.\n"
            "• **Reporting Agent** – implemented by the PDF generator which compiles summary metrics and alerts into a report.\n"
            "This combination automates data, model, monitoring and reporting for the lubricant digital twin."
        )

    # Default fallback answer
    return (
        "This assistant can answer questions about the EcoTwin project, including:\n"
        "• what the project does and how the digital twin works\n"
        "• what RULL means\n"
        "• which dataset and features are used\n"
        "• model performance (RMSE, R²)\n"
        "• how alerts and contamination logic work\n"
        "• how the smart-agent architecture is structured\n\n"
        "Try asking things like:\n"
        "• 'What is this project doing?'\n"
        "• 'What is RULL?'\n"
        "• 'Which features are most important?'\n"
        "• 'How good is the model (RMSE, R2)?'\n"
        "• 'Explain the agents in this system.'"
    )

# ==============================
# STREAMLIT LAYOUT
# ==============================
st.title("🛢️ EcoTwin – Remaining Useful Lubricant Life (RULL) Dashboard")
st.caption("Digital twin using mapped hydraulic system data + RandomForest model and smart-agent behaviours.")

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.header("Dataset & Model Info")
st.sidebar.write(f"📄 Dataset: `{DATA_FILE}`")
st.sidebar.write(f"🧠 Model: `{MODEL_FILE}`")
st.sidebar.write(f"🔢 Rows: {df.shape[0]} | Columns: {df.shape[1]}")

if st.sidebar.button("🔁 Retrain Model (Auto-Agent)"):
    with st.spinner("Retraining model on current dataset..."):
        new_model, rmse_re, r2_re = retrain_model(df, FEATURES, TARGET, MODEL_FILE)
        model = new_model  # update in memory
        st.sidebar.success(f"Retrained! RMSE={rmse_re:.3f}, R²={r2_re:.3f}")

if st.sidebar.button("🧾 Download PDF Report"):
    buffer = generate_pdf_report(df)
    st.sidebar.download_button(
        label="Download EcoTwin Report PDF",
        data=buffer,
        file_name="ecotwin_report.pdf",
        mime="application/pdf"
    )

if st.sidebar.button("📜 Show Alerts Log"):
    if st.session_state.alerts:
        st.sidebar.write("Last Alerts:")
        for a in st.session_state.alerts[-10:]:
            st.sidebar.write(a)
    else:
        st.sidebar.info("No alerts logged yet.")

# ==============================
# TABS
# ==============================
tab_overview, tab_live, tab_manual, tab_explain, tab_explore, tab_assistant = st.tabs(
    ["📊 Overview", "📡 Live Monitor", "🎛️ What-If", "🧠 Explainability", "🔍 Dataset Exploration", "🤖 Assistant"]
)

# ------------------------------
# TAB 1: OVERVIEW
# ------------------------------
with tab_overview:
    st.subheader("Overall Health Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average RULL", f"{df['rull'].mean():.2f}")
    col2.metric("Min RULL", f"{df['rull'].min():.2f}")
    col3.metric("Max RULL", f"{df['rull'].max():.2f}")

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**RULL Distribution** (full dataset)")
        st.bar_chart(df["rull"])

    with col5:
        st.markdown("**Temperature vs RULL** (full dataset)")
        st.scatter_chart(df, x="temp", y="rull")

    st.markdown("**Preview of Dataset**")
    st.dataframe(df.head(10))

# ------------------------------
# TAB 2: LIVE MONITOR (MONITORING AGENT)
# ------------------------------
with tab_live:
    st.subheader("Live Monitoring Simulation")

    st.write("Simulates real-time readings from the dataset and predicts RULL.")

    mode = st.radio("Streaming mode", ["Random cycles", "Sequential index"], horizontal=True)
    auto_stream = st.checkbox("Auto-stream (10 cycles)", value=False)
    interval = st.slider("Interval between cycles (seconds)", 0.1, 2.0, 0.5, 0.1)

    colA, colB = st.columns([1, 2])

    def process_cycle(idx):
        row = df.loc[idx]
        x = row[FEATURES].to_frame().T
        pred_rull = model.predict(x)[0]
        level, msg, color = classify_status(pred_rull, row["contamination_prob"])

        st.session_state.prediction_history.append(
            {"idx": int(idx), "rull": float(pred_rull)}
        )

        if level == "CRITICAL":
            alert_text = create_alert_text(idx, pred_rull, level, msg)
            st.session_state.alerts.append(alert_text)

        with colA:
            st.markdown(f"**Cycle index:** `{idx}`")
            st.markdown(f"**Predicted RULL:** `{pred_rull:.2f}`")
            st.markdown(
                f"**Status:** <span style='color:{color}; font-weight:bold;'>{level}</span>",
                unsafe_allow_html=True
            )
            st.write(msg)
            st.markdown("---")
            st.write(f"🌡️ Temp: {row['temp']:.2f} °C")
            st.write(f"📈 Vibration: {row['vibration']:.3f} mm/s")
            st.write(f"💨 Pressure: {row['pressure']:.2f} bar")
            st.write(f"⏱️ Runtime: {row['runtime_hours']:.2f} hours")
            st.write(f"🧪 Contamination probability: {row['contamination_prob']:.2f}")

    with colA:
        if not auto_stream:
            if st.button("▶️ Next cycle"):
                if mode == "Random cycles":
                    idx = np.random.randint(0, len(df))
                else:
                    idx = st.session_state.live_index
                    st.session_state.live_index = (st.session_state.live_index + 1) % len(df)
                process_cycle(idx)
        else:
            st.info("Auto-streaming 10 cycles...")
            for _ in range(10):
                if mode == "Random cycles":
                    idx = np.random.randint(0, len(df))
                else:
                    idx = st.session_state.live_index
                    st.session_state.live_index = (st.session_state.live_index + 1) % len(df)
                process_cycle(idx)
                time.sleep(interval)
            st.success("Auto-stream complete.")

    with colB:
        st.markdown("**RULL Prediction History**")
        if st.session_state.prediction_history:
            hist_df = pd.DataFrame(st.session_state.prediction_history)
            st.line_chart(hist_df.set_index("idx")["rull"])
        else:
            st.write("No predictions yet.")

# ------------------------------
# TAB 3: MANUAL WHAT-IF
# ------------------------------
with tab_manual:
    st.subheader("Manual What-If Prediction")

    st.write("Adjust parameters to simulate different lubricant conditions and predict RULL.")

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    temp_val = c1.slider("Temperature (°C)",
                         float(df["temp"].min()),
                         float(df["temp"].max()),
                         float(df["temp"].mean()))
    vib_val = c2.slider("Vibration (mm/s)",
                        float(df["vibration"].min()),
                        float(df["vibration"].max()),
                        float(df["vibration"].mean()))
    pressure_val = c3.slider("Pressure (bar)",
                             float(df["pressure"].min()),
                             float(df["pressure"].max()),
                             float(df["pressure"].mean()))
    load_val = c4.slider("Load (0–1)", 0.0, 1.0, float(df["load"].mean()))
    runtime_val = c5.slider("Runtime (hours)",
                            float(df["runtime_hours"].min()),
                            float(df["runtime_hours"].max()),
                            float(df["runtime_hours"].mean()))
    contamination_val = c6.slider("Contamination (0–1)", 0.0, 1.0,
                                  float(df["contamination"].mean()/2 if df['contamination'].max() > 1 else df['contamination'].mean()))

    oxidation_index_val = max(0.0, (temp_val - 60.0) / 10.0)
    viscosity_drift_val = 0.5*(runtime_val/1000.0) + 0.5*(temp_val/100.0)
    thermal_stress_val = (temp_val**2) / 10000.0

    mech_stress_val = 0
    if vib_val >= 2 and vib_val < 4.5:
        mech_stress_val = 1
    elif vib_val >= 4.5:
        mech_stress_val = 2

    contamination_prob_val = contamination_val

    sample = {
        "temp": temp_val,
        "vibration": vib_val,
        "pressure": pressure_val,
        "load": load_val,
        "runtime_hours": runtime_val,
        "contamination": contamination_val,
        "oxidation_index": oxidation_index_val,
        "viscosity_drift": viscosity_drift_val,
        "thermal_stress": thermal_stress_val,
        "mechanical_stress": mech_stress_val,
        "contamination_prob": contamination_prob_val
    }

    if st.button("🔮 Predict RULL"):
        input_df = pd.DataFrame([sample])
        pred = model.predict(input_df)[0]
        level, msg, color = classify_status(pred, contamination_prob_val)

        st.success(f"Predicted RULL: {pred:.2f}")
        st.markdown(
            f"**Status:** <span style='color:{color}; font-weight:bold;'>{level}</span>",
            unsafe_allow_html=True
        )
        st.write(msg)

        st.markdown("**Derived indicators used in this scenario:**")
        st.json({
            "oxidation_index": oxidation_index_val,
            "viscosity_drift": viscosity_drift_val,
            "thermal_stress": thermal_stress_val,
            "mechanical_stress": mech_stress_val,
            "contamination_prob": contamination_prob_val
        })

# ------------------------------
# TAB 4: EXPLAINABILITY
# ------------------------------
with tab_explain:
    st.subheader("Explainability & Feature Importance")

    if feature_importance_df is not None:
        st.markdown("**Feature Importances (RandomForest)**")
        st.bar_chart(feature_importance_df.set_index("feature")["importance"])
        st.dataframe(feature_importance_df.reset_index(drop=True))
    else:
        st.info("Model does not expose feature_importances_ (not a tree-based model).")

    st.markdown("**Simple Future RULL Trend Forecast**")

    from numpy.polynomial.polynomial import polyfit

    sub = df[["runtime_hours", "rull"]].dropna().sort_values("runtime_hours")
    x = sub["runtime_hours"].values
    y = sub["rull"].values

    if len(x) > 2:
        b, m = polyfit(x, y, 1)  # rull ≈ m*x + b
        future_hours = np.linspace(x.max(), x.max() + 50, 20)
        future_rull = m*future_hours + b
        forecast_df = pd.DataFrame({"runtime_hours": future_hours, "forecast_rull": future_rull})
        st.line_chart(forecast_df.set_index("runtime_hours"))
        st.caption("Linear trend forecast of RULL for the next ~50 hours.")
    else:
        st.info("Not enough data points to build a forecast.")

# ------------------------------
# TAB 5: DATASET EXPLORATION
# ------------------------------
with tab_explore:
    st.subheader("Dataset Exploration")

    colX, colY = st.columns(2)

    with colX:
        st.markdown("**Temperature over cycles**")
        temp_series = df[["runtime_hours", "temp"]].sort_values("runtime_hours")
        st.line_chart(temp_series.set_index("runtime_hours"))

    with colY:
        st.markdown("**Vibration over cycles**")
        vib_series = df[["runtime_hours", "vibration"]].sort_values("runtime_hours")
        st.line_chart(vib_series.set_index("runtime_hours"))

    st.markdown("**Full Correlation Matrix**")
    corr_cols = [
        "temp","vibration","pressure","load","runtime_hours",
        "contamination","oxidation_index","viscosity_drift",
        "thermal_stress","mechanical_stress","contamination_prob","rull"
    ]
    corr_matrix = df[corr_cols].corr()
    st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"))

# ------------------------------
# TAB 6: ASSISTANT AGENT
# ------------------------------
with tab_assistant:
    st.subheader("🤖 EcoTwin Assistant Agent")

    st.write("Ask questions about the project, model, data, agents, or alerts.")

    user_q = st.text_area("Type your question about this project here:", height=100, placeholder="Examples: What is RULL? How good is the model? Explain the agents in this system.")
    if st.button("Ask EcoTwin Agent"):
        if not user_q.strip():
            st.warning("Please type a question first.")
        else:
            answer = answer_user_query(user_q)
            st.markdown("**Assistant Response:**")
            st.write(answer)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
import sys
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

# =========================================
# AI REASONING AGENT IMPORTS (PATH FIX)
# =========================================
PROJECT_ROOT = os.getcwd()
AGENT_PATH = os.path.join(PROJECT_ROOT, "agent")
sys.path.insert(0, AGENT_PATH)

from embedder import KnowledgeEmbedder
from retriever import KnowledgeRetriever
from reasoning_agent import ask_agent

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

df = load_data(DATA_FILE)
model = load_model(MODEL_FILE)

# ==============================
# INIT SESSION STATE
# ==============================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ==============================
# INIT AI AGENT (CACHED)
# ==============================
from kb_loader import load_knowledge_base

@st.cache_resource
def load_ai_agent():
    # Load original knowledge documents
    documents = load_knowledge_base(
        kb_path=os.path.join(PROJECT_ROOT, "knowledge")
    )

    embedder = KnowledgeEmbedder()
    embedder.load(
        path=os.path.join(AGENT_PATH, "kb.index"),
        documents=documents
    )

    return KnowledgeRetriever(embedder)

agent_retriever = load_ai_agent()

# ==============================
# HELPER FUNCTIONS
# ==============================
def classify_status(pred_rull, contamination_prob):
    if pred_rull <= RULL_CRITICAL or contamination_prob > 0.7:
        return "CRITICAL", "🚨 Immediate regeneration required!", "red"
    elif pred_rull <= RULL_WARN:
        return "WARNING", "⚠️ Plan regeneration soon.", "orange"
    return "OK", "✅ Lubricant condition acceptable.", "green"

def create_alert_text(idx, pred_rull, level):
    return f"[{datetime.now()}] Cycle {idx} | {level} | RULL={pred_rull:.2f}"

# ==============================
# MODEL METRICS
# ==============================
@st.cache_data
def compute_metrics():
    X = df[FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = model.predict(X_test)
    return {
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds)
    }

metrics_info = compute_metrics()

# ==============================
# UI LAYOUT
# ==============================
st.title("🛢️ EcoTwin – Remaining Useful Lubricant Life (RULL)")
st.caption("AI-powered digital twin for circular lubricant economy")

tabs = st.tabs([
    "📊 Overview",
    "📡 Live Monitor",
    "🎛️ What-If",
    "🧠 Explainability",
    "🔍 Dataset",
    "🤖 AI Agent"
])

# ==============================
# TAB 1: OVERVIEW
# ==============================
with tabs[0]:
    st.metric("Average RULL", f"{df['rull'].mean():.2f}")
    st.bar_chart(df["rull"])

# ==============================
# TAB 2: LIVE MONITOR
# ==============================
with tabs[1]:
    if st.button("▶️ Simulate Cycle"):
        idx = np.random.randint(0, len(df))
        row = df.loc[idx]
        pred = model.predict(row[FEATURES].to_frame().T)[0]
        level, msg, color = classify_status(pred, row["contamination_prob"])

        if level == "CRITICAL":
            st.session_state.alerts.append(create_alert_text(idx, pred, level))

        st.markdown(f"**Predicted RULL:** {pred:.2f}")
        st.markdown(f"**Status:** <span style='color:{color}'>{level}</span>", unsafe_allow_html=True)
        st.write(msg)

# ==============================
# TAB 3: WHAT-IF
# ==============================
with tabs[2]:
    temp = st.slider("Temperature", 20.0, 120.0, 60.0)
    vib = st.slider("Vibration", 0.0, 10.0, 2.0)

    sample = df[FEATURES].mean().to_dict()
    sample["temp"] = temp
    sample["vibration"] = vib

    if st.button("🔮 Predict"):
        pred = model.predict(pd.DataFrame([sample]))[0]
        st.success(f"Predicted RULL: {pred:.2f}")

# ==============================
# TAB 4: EXPLAINABILITY
# ==============================
with tabs[3]:
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": FEATURES,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        st.bar_chart(imp.set_index("feature"))

# ==============================
# TAB 5: DATASET
# ==============================
with tabs[4]:
    st.dataframe(df.head(20))
    st.write(df.describe())

# ==============================
# TAB 6: AI REASONING AGENT
# ==============================
with tabs[5]:
    st.subheader("🤖 EcoTwin AI Reasoning Agent")

    question = st.text_area(
        "Ask anything about the project:",
        height=120,
        placeholder="How does EcoTwin support circular economy?"
    )

    if st.button("Ask Agent"):
        live_context = {
            "average_rull": float(df["rull"].mean()),
            "min_rull": float(df["rull"].min()),
            "max_rull": float(df["rull"].max()),
            "recent_alerts": st.session_state.alerts[-3:],
            "model_rmse": metrics_info["rmse"],
            "model_r2": metrics_info["r2"],
            "total_cycles": len(df)
        }

        with st.spinner("Thinking..."):
            answer, sources = ask_agent(question, agent_retriever, live_context)

        st.markdown("### 🧠 Answer")
        st.write(answer)

        with st.expander("📚 Sources"):
            for s in sources:
                st.write(f"- {s['source']}")

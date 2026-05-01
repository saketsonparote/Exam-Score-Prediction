import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exam Score Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }

    /* Header */
    .hero-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    /* Card wrapper */
    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(167,139,250,0.3);
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, rgba(167,139,250,0.2), rgba(96,165,250,0.2));
        border: 2px solid rgba(167,139,250,0.5);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-score {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .result-label {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    .grade-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 0.8rem;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
        margin-top: 0.5rem;
    }
    div.stButton > button:hover {
        opacity: 0.88;
    }

    /* Slider label */
    label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #f0f0f0 !important;
    }

    /* Slider track */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    }

    /* Metric tiles */
    .metric-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .metric-tile {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 0.8rem 1.4rem;
        text-align: center;
        min-width: 130px;
    }
    .metric-tile .mt-val {
        font-size: 1.4rem;
        font-weight: 800;
        color: #a78bfa;
    }
    .metric-tile .mt-key {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* Tips */
    .tip-box {
        background: rgba(52,211,153,0.1);
        border-left: 4px solid #34d399;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 0.6rem;
        color: #d1fae5;
        font-size: 0.9rem;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Encoders ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("xgb_tuned_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_artifacts()


# ─── Helper: Grade + Colour ──────────────────────────────────────────────────
def get_grade(score):
    if score >= 90:
        return "A+", "#34d399", "🏆 Outstanding! Topper material hai tu!"
    elif score >= 80:
        return "A", "#60a5fa", "🎯 Excellent! Solid performance!"
    elif score >= 70:
        return "B", "#a78bfa", "👍 Good work! Thoda aur push karo!"
    elif score >= 60:
        return "C", "#fbbf24", "📚 Average. Study hours badhao bhai."
    elif score >= 50:
        return "D", "#f97316", "⚠️ Below average. Isko seriously lo."
    else:
        return "F", "#f87171", "🚨 Fail zone. Ab toh padho yaar!"


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🎓 Exam Score Predictor</h1>
    <p>Fill in your study habits and get your predicted exam score instantly</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Input Form ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Study Habit Inputs</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "📖 Daily Study Hours",
        min_value=0.0, max_value=8.0, value=3.0, step=0.1,
        help="Average hours you study per day"
    )

    class_attendance = st.slider(
        "🏫 Class Attendance (%)",
        min_value=40.0, max_value=100.0, value=75.0, step=0.5,
        help="Your overall class attendance percentage"
    )

    sleep_hours = st.slider(
        "😴 Sleep Hours per Night",
        min_value=4.0, max_value=10.0, value=7.0, step=0.1,
        help="Average hours of sleep you get per night"
    )

with col2:
    sleep_quality = st.selectbox(
        "🌙 Sleep Quality",
        options=["good", "average", "poor"],
        help="How well do you sleep?"
    )

    study_method = st.selectbox(
        "📝 Study Method",
        options=["self-study", "group study", "online videos", "coaching", "mixed"],
        help="Your primary method of studying"
    )

    facility_rating = st.selectbox(
        "🏛️ Facility Rating",
        options=["high", "medium", "low"],
        help="How would you rate your study facilities?"
    )

st.markdown("")

# ─── Predict Button ──────────────────────────────────────────────────────────
predict_clicked = st.button("🔮 Predict My Exam Score")

# ─── Prediction Logic ─────────────────────────────────────────────────────────
if predict_clicked:
    # Encode categoricals
    sleep_q_enc = encoders["sleep_quality"].transform([sleep_quality])[0]
    study_m_enc = encoders["study_method"].transform([study_method])[0]
    facility_enc = encoders["facility_rating"].transform([facility_rating])[0]

    # Build input df — same column order as training
    input_df = pd.DataFrame({
        "study_hours":       [study_hours],
        "class_attendance":  [class_attendance],
        "sleep_hours":       [sleep_hours],
        "sleep_quality":     [sleep_q_enc],
        "study_method":      [study_m_enc],
        "facility_rating":   [facility_enc],
    })

    predicted_score = float(model.predict(input_df)[0])
    predicted_score = round(np.clip(predicted_score, 0, 100), 1)

    grade, color, message = get_grade(predicted_score)

    # Result Box
    st.markdown(f"""
    <div class="result-box">
        <div class="result-score">{predicted_score}</div>
        <div class="result-label">Predicted Exam Score (out of 100)</div>
        <span class="grade-badge" style="background: {color}22; border: 2px solid {color}; color: {color};">
            Grade: {grade}
        </span>
        <div style="margin-top:1rem; font-size:1.1rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)

    # Input Summary Tiles
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Input Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="mt-val">{study_hours}h</div>
            <div class="mt-key">Study Hours</div>
        </div>
        <div class="metric-tile">
            <div class="mt-val">{class_attendance}%</div>
            <div class="mt-key">Attendance</div>
        </div>
        <div class="metric-tile">
            <div class="mt-val">{sleep_hours}h</div>
            <div class="mt-key">Sleep Hours</div>
        </div>
        <div class="metric-tile">
            <div class="mt-val">{sleep_quality.capitalize()}</div>
            <div class="mt-key">Sleep Quality</div>
        </div>
        <div class="metric-tile">
            <div class="mt-val">{study_method.title()}</div>
            <div class="mt-key">Study Method</div>
        </div>
        <div class="metric-tile">
            <div class="mt-val">{facility_rating.capitalize()}</div>
            <div class="mt-key">Facility</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Personalised Tips
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 Personalised Tips</div>', unsafe_allow_html=True)

    tips = []
    if study_hours < 2.5:
        tips.append("📖 <b>Study hours kam hai bhai</b> — try to hit at least 3–4 hours daily for a noticeable score bump.")
    if class_attendance < 75:
        tips.append("🏫 <b>Attendance 75% se neeche hai</b> — missing classes directly hurts your score. Show up!")
    if sleep_quality == "poor":
        tips.append("🌙 <b>Poor sleep = poor retention</b> — fix your sleep schedule, no cap.")
    if sleep_hours < 6:
        tips.append("😴 <b>6 ghante se kam so raha hai?</b> — Brain consolidates memory during sleep. Sleep more.")
    if facility_rating == "low":
        tips.append("🏛️ <b>Low facilities?</b> — Compensate with online resources. YouTube + notes = free upgrade.")
    if not tips:
        tips.append("✅ <b>Great inputs!</b> — Keep these habits consistent and you'll crush it.")

    for tip in tips:
        st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#475569; font-size:0.8rem;">Built with XGBoost · Streamlit · 🤙</p>',
    unsafe_allow_html=True
)
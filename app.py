import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor — M. Asad Awan",
    page_icon="💼",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

/* ── App background ── */
.stApp {
    background-color: #0d0d0d;
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Header ── */
.header-wrap {
    border-bottom: 1px solid #1f1f1f;
    padding: 36px 0 28px;
    text-align: center;
    margin-bottom: 32px;
}
.header-wrap h1 {
    color: #f1f1f1;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.header-wrap .sub {
    color: #6b7280;
    font-size: 0.88rem;
    margin: 0 0 16px;
}
.badge-row {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 12px;
}
.badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    color: #a3a3a3;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 6px;
    letter-spacing: 0.3px;
}
.badge.accent {
    background: #18181b;
    border-color: #7c3aed;
    color: #a78bfa;
}

/* ── Section label ── */
.section-label {
    color: #9ca3af;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 0 0 14px;
}

/* ── Form panel ── */
.form-panel {
    background: #111111;
    border: 1px solid #1f1f1f;
    border-radius: 14px;
    padding: 28px 28px 20px;
    margin-bottom: 16px;
}

/* ── Streamlit label overrides ── */
label,
.stSelectbox label,
.stNumberInput label,
.stSlider label {
    color: #9ca3af !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    margin-bottom: 4px !important;
}

/* ── Input / select fields ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #f1f1f1 !important;
    font-size: 0.9rem !important;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:focus {
    border-color: #7c3aed !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: #7c3aed !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 13px 0 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    transition: background 0.2s, transform 0.1s;
    margin-top: 8px;
}
.stButton > button:hover {
    background: #6d28d9 !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Result card ── */
.result-card {
    background: #111111;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #7c3aed;
    border-radius: 14px;
    padding: 32px 28px;
    text-align: center;
    margin-top: 12px;
    animation: fadeUp 0.35s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-card .rlabel {
    color: #6b7280;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 10px;
}
.result-card .salary {
    color: #f1f1f1;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -1.5px;
    line-height: 1;
}
.result-card .model-note {
    color: #6b7280;
    font-size: 0.78rem;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid #1f1f1f;
}
.result-card .model-note strong {
    color: #a78bfa;
}

/* ── Error box ── */
.err-box {
    background: #1a0a0a;
    border: 1px solid #4b1111;
    border-radius: 10px;
    padding: 16px 20px;
    color: #f87171;
    font-size: 0.88rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #3f3f46;
    font-size: 0.72rem;
    margin-top: 48px;
    padding-bottom: 24px;
    border-top: 1px solid #1a1a1a;
    padding-top: 20px;
}

/* ── Metric overrides ── */
[data-testid="metric-container"] {
    background: #111111;
    border: 1px solid #1f1f1f;
    border-radius: 10px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
[data-testid="stMetricValue"] {
    color: #f1f1f1 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_FILE = "salary_model_full_dataset.pkl"

JOB_TITLES = [
    "Account Manager", "Accountant", "Administrative Assistant",
    "Business Analyst", "Business Development Manager", "Business Intelligence Analyst",
    "CEO", "Chief Data Officer", "Chief Technology Officer",
    "Content Marketing Manager", "Copywriter", "Creative Director",
    "Customer Service Manager", "Customer Service Rep", "Customer Service Representative",
    "Customer Success Manager", "Customer Success Rep",
    "Data Analyst", "Data Entry Clerk", "Data Scientist",
    "Digital Content Producer", "Digital Marketing Manager",
    "Director", "Director of Business Development", "Director of Engineering",
    "Director of Finance", "Director of HR", "Director of Human Capital",
    "Director of Human Resources", "Director of Marketing", "Director of Operations",
    "Director of Product Management", "Director of Sales", "Director of Sales and Marketing",
    "Event Coordinator", "Financial Advisor", "Financial Analyst", "Financial Manager",
    "Graphic Designer", "HR Generalist", "HR Manager", "Help Desk Analyst",
    "Human Resources Director", "IT Manager", "IT Support", "IT Support Specialist",
    "Junior Account Manager", "Junior Accountant", "Junior Advertising Coordinator",
    "Junior Business Analyst", "Junior Business Development Associate",
    "Junior Business Operations Analyst", "Junior Copywriter",
    "Junior Customer Support Specialist", "Junior Data Analyst", "Junior Data Scientist",
    "Junior Designer", "Junior Developer", "Junior Financial Advisor",
    "Junior Financial Analyst", "Junior HR Coordinator", "Junior HR Generalist",
    "Junior Marketing Analyst", "Junior Marketing Coordinator", "Junior Marketing Manager",
    "Junior Marketing Specialist", "Junior Operations Analyst", "Junior Operations Coordinator",
    "Junior Operations Manager", "Junior Product Manager", "Junior Project Manager",
    "Junior Recruiter", "Junior Research Scientist", "Junior Sales Representative",
    "Junior Social Media Manager", "Junior Social Media Specialist",
    "Junior Software Developer", "Junior Software Engineer", "Junior UX Designer",
    "Junior Web Designer", "Junior Web Developer",
    "Marketing Analyst", "Marketing Coordinator", "Marketing Manager", "Marketing Specialist",
    "Network Engineer", "Office Manager", "Operations Analyst", "Operations Director",
    "Operations Manager", "Principal Engineer", "Principal Scientist",
    "Product Designer", "Product Manager", "Product Marketing Manager",
    "Project Engineer", "Project Manager", "Public Relations Manager",
    "Recruiter", "Research Director", "Research Scientist",
    "Sales Associate", "Sales Director", "Sales Executive", "Sales Manager",
    "Sales Operations Manager", "Sales Representative",
    "Senior Account Executive", "Senior Account Manager", "Senior Accountant",
    "Senior Business Analyst", "Senior Business Development Manager", "Senior Consultant",
    "Senior Data Analyst", "Senior Data Engineer", "Senior Data Scientist",
    "Senior Engineer", "Senior Financial Advisor", "Senior Financial Analyst",
    "Senior Financial Manager", "Senior Graphic Designer",
    "Senior HR Generalist", "Senior HR Manager", "Senior HR Specialist",
    "Senior Human Resources Coordinator", "Senior Human Resources Manager",
    "Senior Human Resources Specialist", "Senior IT Consultant",
    "Senior IT Project Manager", "Senior IT Support Specialist",
    "Senior Manager", "Senior Marketing Analyst", "Senior Marketing Coordinator",
    "Senior Marketing Director", "Senior Marketing Manager", "Senior Marketing Specialist",
    "Senior Operations Analyst", "Senior Operations Coordinator", "Senior Operations Manager",
    "Senior Product Designer", "Senior Product Development Manager", "Senior Product Manager",
    "Senior Product Marketing Manager", "Senior Project Coordinator", "Senior Project Manager",
    "Senior Quality Assurance Analyst", "Senior Research Scientist", "Senior Researcher",
    "Senior Sales Manager", "Senior Sales Representative", "Senior Scientist",
    "Senior Software Architect", "Senior Software Developer", "Senior Software Engineer",
    "Senior Training Specialist", "Senior UX Designer",
    "Social Media Manager", "Social Media Specialist",
    "Software Developer", "Software Engineer", "Software Manager", "Software Project Manager",
    "Strategy Consultant", "Supply Chain Analyst", "Supply Chain Manager",
    "Technical Recruiter", "Technical Support Specialist", "Technical Writer",
    "Training Specialist", "UX Designer", "UX Researcher",
    "VP of Finance", "VP of Operations", "Web Developer",
]

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <h1>💼 Salary Predictor</h1>
    <p class="sub">Predict employee salary using machine learning</p>
    <div class="badge-row">
        <span class="badge accent">Assignment 3 — Task 2 &nbsp;|&nbsp; AIC354 Machine Learning</span>
    </div>
    <div class="badge-row" style="margin-top:8px;">
        <span class="badge">👤 M. Asad Awan</span>
        <span class="badge">🎓 FA24-BSE-073</span>
        <span class="badge">🏛️ COMSATS University, Lahore</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model missing warning ───────────────────────────────────────────────────────
if model is None:
    st.markdown("""
    <div class="err-box">
        ⚠️ <strong>Model file not found.</strong><br>
        Run the notebook to generate <code>salary_model_full_dataset.pkl</code>,
        then add it to this repo and redeploy.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Employee Profile</p>', unsafe_allow_html=True)
st.markdown('<div class="form-panel">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox(
        "Education Level",
        ["Bachelor's", "Master's", "PhD"],
    )

with col2:
    years_exp = st.number_input(
        "Years of Experience", min_value=0, max_value=40, value=5, step=1
    )
    job_title = st.selectbox("Job Title", JOB_TITLES, index=JOB_TITLES.index("Software Engineer"))

st.markdown("</div>", unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
predict_clicked = st.button("Predict Salary →")

if predict_clicked:
    input_data = pd.DataFrame([{
        "Age"                 : age,
        "Years of Experience" : years_exp,
        "Gender"              : gender,
        "Education Level"     : education,
        "Job Title"           : job_title,
    }])

    input_data = input_data[["Age", "Years of Experience", "Gender", "Education Level", "Job Title"]]

    predicted = model.predict(input_data)[0]
    predicted = max(predicted, 0)

    algo_name = model.named_steps["regressor"].__class__.__name__

    st.markdown(f"""
    <div class="result-card">
        <div class="rlabel">Predicted Annual Salary</div>
        <div class="salary">${predicted:,.0f}</div>
        <div class="model-note">
            Model: <strong>{algo_name}</strong> &nbsp;·&nbsp;
            Evaluated with 10-Fold Cross Validation on the full dataset
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Age", age)
    c2.metric("Experience", f"{years_exp} yrs")
    c3.metric("Education", education)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit &nbsp;·&nbsp; Model: Ridge / Lasso / ElasticNet / SVR / MLP &nbsp;·&nbsp; 10-Fold CV<br>
    M. Asad Awan &nbsp;|&nbsp; FA24-BSE-073 &nbsp;|&nbsp; COMSATS University Islamabad, Lahore Campus
</div>
""", unsafe_allow_html=True)

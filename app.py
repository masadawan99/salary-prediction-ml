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

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Header card */
.header-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 32px 36px 24px;
    margin-bottom: 28px;
    backdrop-filter: blur(12px);
    text-align: center;
}
.header-card h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.header-card .subtitle {
    color: rgba(255,255,255,0.55);
    font-size: 0.9rem;
    margin: 0;
}
.badge {
    display: inline-block;
    background: linear-gradient(90deg, #6c63ff, #a78bfa);
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 20px;
    margin-top: 12px;
    letter-spacing: 0.4px;
}

/* Form card */
.form-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 30px 32px;
    backdrop-filter: blur(10px);
    margin-bottom: 24px;
}
.form-card h3 {
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 20px;
    letter-spacing: 0.3px;
}

/* Streamlit label override */
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Input fields */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #6c63ff, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    transition: opacity 0.2s;
    margin-top: 8px;
}
.stButton > button:hover { opacity: 0.88; }

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(167,139,250,0.15));
    border: 1px solid rgba(108,99,255,0.5);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    backdrop-filter: blur(12px);
    margin-top: 8px;
    animation: fadeUp 0.4s ease;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-card .label {
    color: rgba(255,255,255,0.6);
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.result-card .salary {
    color: #ffffff;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1.1;
}
.result-card .rmse-note {
    color: rgba(255,255,255,0.45);
    font-size: 0.78rem;
    margin-top: 10px;
}

/* Info pills */
.pill-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 12px;
    justify-content: center;
}
.pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.7);
    font-size: 0.76rem;
    padding: 5px 14px;
    border-radius: 20px;
}

/* Error */
.err-box {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 12px;
    padding: 16px 20px;
    color: #fca5a5;
    font-size: 0.88rem;
}

/* Footer */
.footer {
    text-align: center;
    color: rgba(255,255,255,0.3);
    font-size: 0.75rem;
    margin-top: 40px;
    padding-bottom: 20px;
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
<div class="header-card">
    <h1>💼 Salary Predictor</h1>
    <p class="subtitle">Predict employee salary using machine learning</p>
    <span class="badge">Assignment 3 — Task 2 &nbsp;|&nbsp; AIC354 Machine Learning</span>
    <div class="pill-row">
        <span class="pill">👤 M. Asad Awan</span>
        <span class="pill">🎓 FA24-BSE-073</span>
        <span class="pill">🏛️ COMSATS University, Lahore</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model missing warning ───────────────────────────────────────────────────────
if model is None:
    st.markdown("""
    <div class="err-box">
        ⚠️ <strong>Model file not found.</strong><br>
        Run the notebook on Google Colab to generate
        <code>salary_model_full_dataset.pkl</code>, then add it to this repo and redeploy.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="form-card"><h3>🧑‍💼 Enter Employee Profile</h3>', unsafe_allow_html=True)

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
predict_clicked = st.button("🔮 Predict Salary")

if predict_clicked:
    input_data = pd.DataFrame([{
        "Age"                 : age,
        "Years of Experience" : years_exp,
        "Gender"              : gender,
        "Education Level"     : education,
        "Job Title"           : job_title,
    }])

    # Column order must match training
    input_data = input_data[["Age", "Years of Experience", "Gender", "Education Level", "Job Title"]]

    predicted = model.predict(input_data)[0]
    predicted = max(predicted, 0)  # guard against negative predictions

    algo_name = model.named_steps["regressor"].__class__.__name__

    st.markdown(f"""
    <div class="result-card">
        <div class="label">Predicted Annual Salary</div>
        <div class="salary">${predicted:,.0f}</div>
        <div class="rmse-note">
            Model: <strong>{algo_name}</strong> &nbsp;·&nbsp;
            Evaluated with 10-Fold Cross Validation on the full dataset
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary sidebar-style details
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Age", age)
    c2.metric("Experience", f"{years_exp} yrs")
    c3.metric("Education", education)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit &nbsp;·&nbsp; ML Model: Ridge / Lasso / ElasticNet / SVR / MLP &nbsp;·&nbsp; 10-Fold CV
    <br>M. Asad Awan &nbsp;|&nbsp; FA24-BSE-073 &nbsp;|&nbsp; COMSATS University Islamabad, Lahore Campus
</div>
""", unsafe_allow_html=True)

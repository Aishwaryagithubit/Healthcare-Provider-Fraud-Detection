
from pathlib import Path
import pickle
import warnings
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Healthcare Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "healthcare_fraud_model.pkl"

# ============================================================
# EXACT FEATURE CONTRACT FROM TRAINING PIPELINE
# ============================================================

FEATURE_COLUMNS = [
    "Patient_Age",
    "Patient_Gender",
    "Diagnosis_Code",
    "Procedure_Code",
    "Claim_Amount",
    "Approved_Amount",
    "Insurance_Type",
    "Days_Between_Service_and_Claim",
    "Number_of_Claims_Per_Provider_Monthly",
    "Provider_Specialty",
    "Patient_State",
    "Claim_Status",
    "Length_of_Stay",
    "Visit_Type",
    "Chronic_Condition_Flag",
    "Prior_Visits_12m",
    "Claim_Year",
    "Claim_Month",
    "Claim_Day",
    "Claim_DayOfWeek",
    "Amount_Difference",
    "Approval_Ratio",
]

CATEGORICAL_FEATURES = [
    "Patient_Gender",
    "Diagnosis_Code",
    "Insurance_Type",
    "Provider_Specialty",
    "Patient_State",
    "Claim_Status",
    "Visit_Type",
]

NUMERICAL_FEATURES = [
    "Patient_Age",
    "Procedure_Code",
    "Claim_Amount",
    "Approved_Amount",
    "Days_Between_Service_and_Claim",
    "Number_of_Claims_Per_Provider_Monthly",
    "Length_of_Stay",
    "Chronic_Condition_Flag",
    "Prior_Visits_12m",
    "Claim_Year",
    "Claim_Month",
    "Claim_Day",
    "Claim_DayOfWeek",
    "Amount_Difference",
    "Approval_Ratio",
]

# These are the categories present in the training dataset.
GENDER_OPTIONS = ["Male", "Female"]
DIAGNOSIS_OPTIONS = [
    "I25.10", "E11.9", "J06.9", "I10", "M54.5",
    "E78.5", "K21.9", "F41.9", "N39.0", "J18.9",
]
INSURANCE_OPTIONS = ["Medicaid", "Self-Pay", "Private", "Medicare"]
SPECIALTY_OPTIONS = [
    "Cardiology",
    "General Practice",
    "Pulmonology",
    "Orthopedics",
    "Internal Medicine",
    "Neurology",
]
STATE_OPTIONS = ["NY", "IL", "TX", "PA", "CA", "OH", "FL", "GA"]
STATUS_OPTIONS = ["Approved", "Pending", "Rejected"]
VISIT_OPTIONS = ["Outpatient", "Inpatient", "Emergency"]


# ============================================================
# PREMIUM NAVY / CYAN UI
# ============================================================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #07111f;
        --panel: #0d1b2e;
        --panel-2: #10243b;
        --line: #203b57;
        --muted: #8ea7c0;
        --text: #eef6ff;
        --cyan: #25b8f2;
        --blue: #2563eb;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(37,184,242,.14), transparent 27%),
            radial-gradient(circle at 94% 5%, rgba(37,99,235,.14), transparent 30%),
            #07111f;
        color: var(--text);
        font-family: Inter, sans-serif;
    }

    .main .block-container {
        max-width: 1480px;
        padding: 30px 48px 45px;
    }

    header[data-testid="stHeader"] {
        background: #07111f;
    }

    /* Hero */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 48px 52px 45px;
        border-radius: 28px;
        background:
            radial-gradient(circle at 82% 15%, rgba(37,184,242,.17), transparent 27%),
            linear-gradient(135deg, #0b1729 0%, #10233b 55%, #123b5c 100%);
        border: 1px solid rgba(37,184,242,.27);
        box-shadow: 0 25px 70px rgba(0,0,0,.35);
        margin-bottom: 22px;
    }

    .hero-badge {
        display: inline-block;
        padding: 8px 14px;
        border: 1px solid rgba(37,184,242,.45);
        border-radius: 999px;
        background: rgba(4,47,73,.55);
        color: #39c1f5;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
    }

    .hero-title {
        margin-top: 21px;
        color: #f7fbff;
        font-size: clamp(36px, 4.6vw, 60px);
        font-weight: 800;
        letter-spacing: -2.3px;
        line-height: 1.02;
    }

    .hero-title span {
        color: #27b7ef;
    }

    .hero-subtitle {
        margin-top: 17px;
        max-width: 820px;
        color: #91abc5;
        font-size: 15px;
        line-height: 1.7;
    }

    /* Notice */
    .notice {
        padding: 14px 18px;
        margin: 0 0 27px;
        border: 1px solid rgba(37,184,242,.16);
        border-left: 4px solid #25b8f2;
        border-radius: 11px;
        background: rgba(13,29,48,.92);
        color: #a9bdd1;
        font-size: 13px;
        line-height: 1.55;
    }

    .notice strong { color: #eef8ff; }

    /* Section labels */
    .section-title {
        margin: 25px 0 13px;
        color: #edf6ff;
        font-size: 19px;
        font-weight: 800;
        letter-spacing: -.2px;
    }

    /* Streamlit widgets */
    [data-testid="stWidgetLabel"] p,
    label {
        color: #93abc2 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="input"],
    div[data-baseweb="select"] {
        background: #0c1a2c !important;
        border: 1px solid #223c58 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"]:focus-within {
        border-color: #25b8f2 !important;
        box-shadow: 0 0 0 1px rgba(37,184,242,.20) !important;
    }

    /* ========================================================
       INPUT TEXT VISIBILITY — FORCE WHITE TEXT
       Streamlit/BaseWeb uses several nested elements depending
       on the widget/version, so these selectors intentionally
       cover all input, select and date widgets.
       ======================================================== */
    .stApp input,
    .stApp textarea,
    .stApp select,
    .stApp [data-baseweb="input"] input,
    .stApp [data-baseweb="base-input"] input,
    .stApp [data-testid="stNumberInput"] input,
    .stApp [data-testid="stTextInput"] input,
    .stApp [data-testid="stDateInput"] input,
    .stApp [data-baseweb="select"] input {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #25b8f2 !important;
        opacity: 1 !important;
        background-color: transparent !important;
        text-shadow: none !important;
    }

    /* Number input inner elements */
    .stApp [data-testid="stNumberInput"] *,
    .stApp [data-testid="stDateInput"] * {
        --text-color: #111827 !important;
    }

    .stApp [data-testid="stNumberInput"] input,
    .stApp [data-testid="stDateInput"] input {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    /* Selected value inside Streamlit selectbox */
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"],
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }

    /* Dropdown menu and options */
    .stApp [role="listbox"],
    .stApp [role="listbox"] *,
    .stApp [role="option"],
    .stApp [role="option"] * {
        background-color: #102238 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Date-picker calendar text */
    .stApp [data-baseweb="calendar"] *,
    .stApp [data-baseweb="calendar"] button {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Prevent Chrome autofill from turning text black */
    .stApp input:-webkit-autofill,
    .stApp input:-webkit-autofill:hover,
    .stApp input:-webkit-autofill:focus {
        -webkit-text-fill-color: #ffffff !important;
        -webkit-box-shadow: 0 0 0 1000px #0c1a2c inset !important;
        box-shadow: 0 0 0 1000px #0c1a2c inset !important;
    }

    input::placeholder {
        color: #7189a1 !important;
        -webkit-text-fill-color: #7189a1 !important;
        opacity: 1 !important;
    }

    /* Button */
    .stButton > button {
        min-height: 50px;
        border: 1px solid rgba(37,184,242,.32) !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #104b7a, #155ed0, #1e7ee8) !important;
        color: #fff !important;
        font-weight: 800 !important;
        box-shadow: 0 13px 32px rgba(21,94,208,.28);
    }

    .stButton > button:hover {
        border-color: rgba(37,184,242,.75) !important;
        transform: translateY(-1px);
        box-shadow: 0 17px 36px rgba(21,126,232,.34);
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        min-height: 105px;
        padding: 20px 22px;
        border: 1px solid #203b57;
        border-radius: 16px;
        background: linear-gradient(145deg, #102238, #0b192a);
        box-shadow: 0 12px 30px rgba(0,0,0,.22);
    }

    div[data-testid="stMetricLabel"] {
        color: #7f9ab5 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f6fbff !important;
        font-weight: 800 !important;
    }

    /* Results */
    .risk-card {
        padding: 24px 27px;
        margin-top: 18px;
        border-radius: 16px;
        box-shadow: 0 15px 35px rgba(0,0,0,.22);
    }

    .risk-card h3 {
        margin: 0 0 12px;
        font-size: 18px;
    }

    .risk-card p {
        color: #c4d2df;
        line-height: 1.65;
        margin: 7px 0;
    }

    .risk-card strong { color: #fff; }

    .risk-high {
        background: linear-gradient(135deg, rgba(72,24,34,.96), rgba(44,17,27,.96));
        border: 1px solid rgba(248,113,113,.28);
        border-left: 5px solid #ef4444;
    }

    .risk-high h3 { color: #fca5a5; }

    .risk-low {
        background: linear-gradient(135deg, rgba(13,61,50,.96), rgba(10,42,37,.96));
        border: 1px solid rgba(74,222,128,.25);
        border-left: 5px solid #22c55e;
    }

    .risk-low h3 { color: #86efac; }

    /* Expander */
    div[data-testid="stExpander"] {
        background: #0c192b !important;
        border: 1px solid #203b57 !important;
        border-radius: 12px !important;
    }

    /* Progress */
    div[data-testid="stProgress"] > div {
        background: #172b42 !important;
        border-radius: 999px !important;
    }

    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #159bd7, #38c5f5) !important;
        border-radius: 999px !important;
    }

    hr { border-color: #20384f !important; }

    .footer {
        padding: 18px 0 3px;
        text-align: center;
        color: #617a94;
        font-size: 12px;
    }

    @media (max-width: 900px) {
        .main .block-container { padding: 20px 16px 35px; }
        .hero { padding: 34px 27px; border-radius: 21px; }
        .hero-title { font-size: 38px; }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD THE ALREADY-TRAINED PIPELINE
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Expected: model/healthcare_fraud_model.pkl"
    )
    st.stop()

try:
    with open(MODEL_PATH, "rb") as file:
        package = pickle.load(file)

    model = package["model"]
    threshold = float(package["threshold"])

    # Prefer the exact feature contract stored with the trained model.
    saved_features = package.get("feature_columns", FEATURE_COLUMNS)
    if list(saved_features) != FEATURE_COLUMNS:
        st.error("Model feature contract does not match the application pipeline.")
        st.stop()

except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AI-POWERED &nbsp; · &nbsp; HEALTHCARE ANALYTICS &nbsp; · &nbsp; ML SYSTEM</div>
        <div class="hero-title">
            Healthcare Provider<br>
            <span>Fraud Intelligence</span>
        </div>
        <div class="hero-subtitle">
            Machine-learning based claim-level risk screening using the
            same feature engineering and trained ensemble pipeline used
            during model development.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
        <strong>Decision-support notice:</strong>
        This application provides a machine-learning risk signal. It does not
        make a final fraud determination. Higher-risk claims should be reviewed
        by an appropriate human investigator.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PATIENT / CLINICAL INPUTS
# ============================================================

st.markdown(
    '<div class="section-title">👤 Patient & Clinical Information</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    patient_age = st.number_input(
        "Patient age",
        min_value=0,
        max_value=120,
        value=45,
        step=1,
    )

    patient_gender = st.selectbox(
        "Patient gender",
        GENDER_OPTIONS,
        index=1,
    )

    chronic_condition = st.selectbox(
        "Chronic condition",
        ["No", "Yes"],
        index=0,
    )

with c2:
    diagnosis_code = st.selectbox(
        "Diagnosis code",
        DIAGNOSIS_OPTIONS,
        index=3,  # I10
    )

    procedure_code = st.number_input(
        "Procedure code",
        min_value=0,
        value=99213,
        step=1,
    )

    visit_type = st.selectbox(
        "Visit type",
        VISIT_OPTIONS,
        index=0,
    )

with c3:
    length_of_stay = st.number_input(
        "Length of stay",
        min_value=0,
        value=1,
        step=1,
    )

    prior_visits = st.number_input(
        "Prior visits in last 12 months",
        min_value=0.0,
        value=3.0,
        step=1.0,
    )

    provider_specialty = st.selectbox(
        "Provider specialty",
        SPECIALTY_OPTIONS,
        index=0,
    )

# ============================================================
# CLAIM / INSURANCE INPUTS
# ============================================================

st.markdown(
    '<div class="section-title">💳 Claim & Insurance Information</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    claim_amount = st.number_input(
        "Claim amount",
        min_value=0.0,
        value=920.0,
        step=10.0,
        format="%.2f",
    )

    approved_amount = st.number_input(
        "Approved amount",
        min_value=0.0,
        value=850.0,
        step=10.0,
        format="%.2f",
    )

with c2:
    insurance_type = st.selectbox(
        "Insurance type",
        INSURANCE_OPTIONS,
        index=2,  # Private
    )

    claim_status = st.selectbox(
        "Claim status",
        STATUS_OPTIONS,
        index=0,
    )

    patient_state = st.selectbox(
        "Patient state",
        STATE_OPTIONS,
        index=0,
    )

with c3:
    days_between = st.number_input(
        "Days between service and claim",
        min_value=0,
        value=15,
        step=1,
    )

    monthly_claims = st.number_input(
        "Provider claims per month",
        min_value=0,
        value=60,
        step=1,
    )

    submission_date = st.date_input(
        "Claim submission date",
        value=date(2024, 9, 1),
    )

# ============================================================
# EXACT FEATURE ENGINEERING FROM data_preprocessing.py
# ============================================================

def build_model_input():
    """
    Reproduce the feature engineering performed before training.

    Important:
    Approval_Ratio intentionally becomes NaN when Claim_Amount == 0.
    The trained preprocessing pipeline then handles that NaN with its
    median imputer, exactly as during training.
    """

    submission_timestamp = pd.to_datetime(
        submission_date,
        errors="coerce",
    )

    amount_difference = (
        claim_amount - approved_amount
    )

    if claim_amount == 0:
        approval_ratio = np.nan
    else:
        approval_ratio = (
            approved_amount / claim_amount
        )

    input_data = pd.DataFrame(
        [{
            "Patient_Age": patient_age,
            "Patient_Gender": patient_gender,
            "Diagnosis_Code": diagnosis_code,
            "Procedure_Code": procedure_code,
            "Claim_Amount": claim_amount,
            "Approved_Amount": approved_amount,
            "Insurance_Type": insurance_type,
            "Days_Between_Service_and_Claim": days_between,
            "Number_of_Claims_Per_Provider_Monthly": monthly_claims,
            "Provider_Specialty": provider_specialty,
            "Patient_State": patient_state,
            "Claim_Status": claim_status,
            "Length_of_Stay": length_of_stay,
            "Visit_Type": visit_type,
            "Chronic_Condition_Flag": (
                1 if chronic_condition == "Yes" else 0
            ),
            "Prior_Visits_12m": prior_visits,
            "Claim_Year": submission_timestamp.year,
            "Claim_Month": submission_timestamp.month,
            "Claim_Day": submission_timestamp.day,
            "Claim_DayOfWeek": submission_timestamp.dayofweek,
            "Amount_Difference": amount_difference,
            "Approval_Ratio": approval_ratio,
        }],
        columns=FEATURE_COLUMNS,
    )

    return input_data


# ============================================================
# RUN PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Run Risk Assessment</div>',
    unsafe_allow_html=True,
)

if st.button(
    "Assess Claim Risk",
    type="primary",
    use_container_width=True,
):
    input_data = build_model_input()

    try:
        probability = float(
            model.predict_proba(input_data)[0, 1]
        )
    except Exception as exc:
        st.error(
            "Prediction failed. The input could not be processed by the "
            f"trained pipeline: {exc}"
        )
        st.stop()

    prediction = int(
        probability >= threshold
    )

    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Assessment Result</div>',
        unsafe_allow_html=True,
    )

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Fraud probability",
            f"{probability:.1%}",
        )

    with r2:
        st.metric(
            "Decision threshold",
            f"{threshold:.0%}",
        )

    with r3:
        st.metric(
            "Risk level",
            "Review" if prediction else "Lower Risk",
        )

    # ========================================================
    # RISK MESSAGE
    # ========================================================

    if prediction:
        st.markdown(
            f"""
            <div class="risk-card risk-high">
                <h3>⚠️ Potential Fraud Risk</h3>
                <p>
                    The trained ensemble estimates a
                    <strong>{probability:.1%}</strong>
                    probability of fraud for this claim.
                </p>
                <p>
                    The probability is above the model's optimized
                    validation threshold of <strong>{threshold:.0%}</strong>.
                    Consider the claim for further investigation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="risk-card risk-low">
                <h3>✓ Lower Predicted Fraud Risk</h3>
                <p>
                    The trained ensemble estimates a
                    <strong>{probability:.1%}</strong>
                    probability of fraud for this claim.
                </p>
                <p>
                    The probability is below the model's optimized
                    validation threshold of <strong>{threshold:.0%}</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(
        min(max(probability, 0.0), 1.0),
        text=f"Fraud probability: {probability:.1%}",
    )

    # ========================================================
    # INPUT AUDIT VIEW
    # ========================================================

    with st.expander("View engineered claim inputs"):
        display_data = input_data.copy()
        display_data["Approval_Ratio"] = display_data[
            "Approval_Ratio"
        ].round(4)

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Risk scores are model outputs and should be combined with "
        "human investigation and appropriate business rules."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        Healthcare Fraud Intelligence &nbsp;•&nbsp;
        Machine Learning Decision Support
    </div>
    """,
    unsafe_allow_html=True,
)

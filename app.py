
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =====================================================
# 1. PROJECT PATHS
# =====================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
RESULTS_PATH = ROOT / "reports" / "model_comparison.csv"
REPORTS_DIR = ROOT / "reports"

# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CreditWise | Risk Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 3. CUSTOM CSS DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0b1020 0%,
        #151735 50%,
        #101c35 100%
    );
    color: #f1f5f9;
}

.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #ffffff !important;
}

p, label {
    color: #dbe4f0;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #17152e,
        #111827
    );
    border-right: 1px solid #37345c;
}

/* Main banner */

.hero {
    padding: 32px;
    border-radius: 22px;
    background: linear-gradient(
        120deg,
        #312e81,
        #6d28d9,
        #0e7490
    );
    margin-bottom: 24px;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.25);
}

.hero h1 {
    font-size: 38px;
    margin: 0;
    color: white !important;
}

.hero p {
    font-size: 16px;
    margin: 10px 0 0 0;
    color: #e0e7ff;
}

/* Section containers */

.section-card {
    background: rgba(30, 41, 70, 0.70);
    border: 1px solid #414b70;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 18px;
}

/* Metric cards */

[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #202742,
        #292650
    );
    border: 1px solid #414b70;
    border-radius: 16px;
    padding: 20px;
}

[data-testid="stMetricLabel"] {
    color: #c4b5fd !important;
}

[data-testid="stMetricValue"] {
    color: #67e8f9 !important;
    font-weight: 800;
}

/* Input boxes */

.stNumberInput input,
.stTextInput input {
    background-color: #202742 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Dropdowns */

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #202742 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(
        90deg,
        #7c3aed,
        #4f46e5,
        #0891b2
    );
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 18px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    filter: brightness(1.2);
    border: 1px solid #a5b4fc;
}

/* Form */

[data-testid="stForm"] {
    background: rgba(30, 41, 70, 0.50);
    border: 1px solid #414b70;
    padding: 24px;
    border-radius: 18px;
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Progress bar */

.stProgress > div > div > div {
    background: linear-gradient(
        90deg,
        #8b5cf6,
        #22d3ee
    );
}

/* Tables */

[data-testid="stDataFrame"] {
    border: 1px solid #414b70;
    border-radius: 12px;
}

/* Footer */

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 24px 0;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#a78bfa;'>💳 CreditWise</h2>",
        unsafe_allow_html=True
    )

    st.caption("Machine Learning Risk Analytics")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 Risk Prediction",
            "📊 Model Performance",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.markdown("### 🧠 ML Models")

    st.write("• Logistic Regression")
    st.write("• Decision Tree")
    st.write("• Random Forest")

    st.divider()

    st.caption("Academic project | Python + Scikit-learn")


# =====================================================
# 5. LOAD TRAINED MODEL
# =====================================================

if not MODEL_PATH.exists():

    st.markdown("""
    <div class="hero">
        <h1>CreditWise</h1>
        <p>Your machine learning credit-risk analysis dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    st.error(
        "Trained model not found. Run `python train.py` "
        "from the project folder first."
    )

    st.stop()


bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]
model_name = bundle["model_name"]

# =====================================================
# 6. HUMAN-READABLE OPTIONS
# =====================================================

sex_options = {
    "Male": 1,
    "Female": 2
}

education_options = {
    "Graduate school": 1,
    "University": 2,
    "High school": 3,
    "Others": 4
}

marriage_options = {
    "Married": 1,
    "Single": 2,
    "Others": 3
}

payment_options = {
    "No consumption (-2)": -2,
    "Paid duly (-1)": -1,
    "Revolving credit (0)": 0,
    "Delayed by 1 month (1)": 1,
    "Delayed by 2 months (2)": 2,
    "Delayed by 3 months (3)": 3,
    "Delayed by 4 months (4)": 4,
    "Delayed by 5 months (5)": 5,
    "Delayed by 6 months (6)": 6,
    "Delayed by 7 months (7)": 7,
    "Delayed by 8 months (8)": 8
}

# =====================================================
# 7. DASHBOARD PAGE
# =====================================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to CreditWise ✨</h1>
        <p>
            Intelligent credit default risk analysis powered
            by Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📈 Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Records",
            "30,000",
            help="Number of customer records in the original dataset."
        )

    with col2:
        st.metric(
            "Input Features",
            len(features)
        )

    with col3:
        st.metric(
            "Trained Models",
            "3"
        )

    st.markdown("### 🚀 What can you do?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>🔍 Risk Prediction</h3>
            <p>
                Enter customer financial details and explore
                the model's estimated next-month default risk.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>📊 Model Performance</h3>
            <p>
                Compare trained models using Accuracy,
                Precision, Recall, F1-score and ROC-AUC.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🤖 Current Best Model")

    st.success(f"Selected model: {model_name}")

    st.info(
        "This project estimates next-month credit-card payment "
        "default using historical data. It does not determine "
        "a person's complete creditworthiness."
    )


# =====================================================
# 8. RISK PREDICTION PAGE
# =====================================================

elif page == "🔍 Risk Prediction":

    st.markdown("""
    <div class="hero">
        <h1>Customer Risk Analysis 🔍</h1>
        <p>
            Enter financial and repayment details
            to generate a model prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Sample values are provided to help you test the application. "
        "Replace them with appropriate example values as needed."
    )

    with st.form("customer_form"):

        # Personal details

        st.markdown("### 👤 Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Customer age",
                min_value=18,
                max_value=100,
                value=30,
                step=1
            )

        with col2:
            sex = st.selectbox(
                "Sex",
                list(sex_options.keys())
            )

        with col3:
            education = st.selectbox(
                "Education",
                list(education_options.keys())
            )

        marriage = st.selectbox(
            "Marital status",
            list(marriage_options.keys())
        )

        st.divider()

        # Credit limit

        st.markdown("### 💰 Credit Information")

        limit_bal = st.number_input(
            "Credit limit",
            min_value=0,
            value=50000,
            step=5000
        )

        st.divider()

        # Repayment history

        st.markdown("### 📅 Repayment History")

        st.caption(
            "Select the repayment-status category for each "
            "historical period. These options follow the dataset's "
            "original codes."
        )

        pay_values = {}

        pay_columns = [
            ("PAY_0", "Most recent period"),
            ("PAY_2", "Previous period 1"),
            ("PAY_3", "Previous period 2"),
            ("PAY_4", "Previous period 3"),
            ("PAY_5", "Previous period 4"),
            ("PAY_6", "Previous period 5")
        ]

        cols = st.columns(3)

        for i, (column, label) in enumerate(pay_columns):

            with cols[i % 3]:

                selected = st.selectbox(
                    label,
                    list(payment_options.keys()),
                    index=2,
                    key=column
                )

                pay_values[column] = payment_options[selected]

        st.divider()

        # Bill amounts

        st.markdown("### 🧾 Monthly Bill Amounts")

        bill_values = {}

        bill_columns = [
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6"
        ]

        default_bills = [
            20000, 18000, 16000,
            14000, 12000, 10000
        ]

        cols = st.columns(3)

        for i, column in enumerate(bill_columns):

            with cols[i % 3]:

                bill_values[column] = st.number_input(
                    f"Bill amount {i + 1}",
                    min_value=0,
                    value=default_bills[i],
                    step=1000,
                    key=column
                )

        st.divider()

        # Payment amounts

        st.markdown("### 💵 Monthly Payment Amounts")

        payment_amounts = {}

        amount_columns = [
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6"
        ]

        cols = st.columns(3)

        for i, column in enumerate(amount_columns):

            with cols[i % 3]:

                payment_amounts[column] = st.number_input(
                    f"Payment amount {i + 1}",
                    min_value=0,
                    value=2000,
                    step=500,
                    key=column
                )

        st.divider()

        submitted = st.form_submit_button(
            "✨ Predict Default Risk",
            type="primary",
            use_container_width=True
        )

    # Prediction output

    if submitted:

        customer_data = {
            "LIMIT_BAL": limit_bal,
            "SEX": sex_options[sex],
            "EDUCATION": education_options[education],
            "MARRIAGE": marriage_options[marriage],
            "AGE": age,
            **pay_values,
            **bill_values,
            **payment_amounts
        }

        input_df = pd.DataFrame(
            [[customer_data[column] for column in features]],
            columns=features
        )

        prediction = int(model.predict(input_df)[0])

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        st.markdown("---")
        st.markdown("## 📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Estimated Default Probability",
                f"{probability:.1%}"
            )

        with col2:

            st.metric(
                "Estimated Non-default Probability",
                f"{1 - probability:.1%}"
            )

        st.progress(probability)

        if prediction == 1:

            st.error(
                "Model output: Default class (1). "
                "The model classifies this input as the default class."
            )

        else:

            st.success(
                "Model output: Non-default class (0). "
                "The model classifies this input as the non-default class."
            )

        st.caption(
            "Probabilities and classes are model estimates, not "
            "guarantees. This educational tool should not be used "
            "alone to make actual lending or credit-approval decisions."
        )


# =====================================================
# 9. MODEL PERFORMANCE PAGE
# =====================================================

elif page == "📊 Model Performance":

    st.markdown("""
    <div class="hero">
        <h1>Model Performance 📊</h1>
        <p>
            Compare the performance of the three trained
            machine learning models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if RESULTS_PATH.exists():

        results = pd.read_csv(RESULTS_PATH)

        st.markdown("### 🏆 Model Comparison")

        st.dataframe(
            results.style.format({
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-score": "{:.3f}",
                "ROC-AUC": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        if "F1-score" in results.columns:

            best_row = results.loc[
                results["F1-score"].idxmax()
            ]

            st.success(
                f"Highest F1-score in this comparison: "
                f"{best_row['Model']} "
                f"({best_row['F1-score']:.3f})"
            )

        st.caption(
            "The saved model is selected by F1-score in the training "
            "script. Accuracy alone may be misleading when classes "
            "are imbalanced."
        )

    else:

        st.warning(
            "Model comparison file not found. Run `python train.py` "
            "to generate the evaluation results."
        )

    # Evaluation graphs

    st.markdown("### 📈 Evaluation Charts")

    confusion_path = REPORTS_DIR / "confusion_matrix.png"
    roc_path = REPORTS_DIR / "roc_curve.png"
    target_path = REPORTS_DIR / "target_distribution.png"

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Confusion Matrix")

        if confusion_path.exists():

            st.image(
                str(confusion_path),
                use_container_width=True
            )

        else:

            st.info(
                "Confusion matrix not found. Run `python train.py`."
            )

    with col2:

        st.markdown("#### ROC Curve")

        if roc_path.exists():

            st.image(
                str(roc_path),
                use_container_width=True
            )

        else:

            st.info(
                "ROC curve not found. Run `python train.py`."
            )

    if target_path.exists():

        st.markdown("#### Dataset Target Distribution")

        st.image(
            str(target_path),
            use_container_width=True
        )


# =====================================================
# 10. ABOUT PROJECT PAGE
# =====================================================

elif page == "ℹ️ About Project":

    st.markdown("""
    <div class="hero">
        <h1>About CreditWise 💡</h1>
        <p>
            A student machine learning project for exploring
            credit-card payment default prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Project Objective")

    st.write(
        "Build and compare machine learning models that estimate "
        "whether a customer will default on a credit-card payment "
        "in the following month."
    )

    st.markdown("### 🛠️ Technologies Used")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="section-card">
            <h3>🐍 Python</h3>
            <p>Core programming language</p>
            <h3>📦 Pandas & NumPy</h3>
            <p>Data processing and analysis</p>
            <h3>🤖 Scikit-learn</h3>
            <p>Machine learning and evaluation</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="section-card">
            <h3>📊 Matplotlib</h3>
            <p>Evaluation charts</p>
            <h3>🌐 Streamlit</h3>
            <p>Interactive web application</p>
            <h3>💾 Joblib</h3>
            <p>Saving and loading the trained model</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧪 Models Compared")

    st.write("1. Logistic Regression")
    st.write("2. Decision Tree Classifier")
    st.write("3. Random Forest Classifier")

    st.markdown("### ⚠️ Limitations")

    st.warning(
        "The dataset concerns next-month credit-card payment default. "
        "It does not capture every aspect of creditworthiness. "
        "The model may reflect limitations or biases in its training "
        "data and is intended for educational exploration only."
    )


# =====================================================
# 11. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    CreditWise | Machine Learning Project<br>
    Built with Python, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)

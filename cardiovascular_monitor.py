# cardiovascular_monitor.py
"""
 - best_heart_model.pkl  (Pipeline including preprocessing)
 - model_comparison.csv  (optional, produced by training script)
 - heart.csv             (original dataset)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# --------------------------
# Config and constants
# --------------------------
MODEL_FILE = "best_heart_model.pkl"
COMPARISON_CSV = "model_performance.csv"
DATASET_CSV = "heart.csv"
HISTORY_CSV = "pred_history.csv"

# The numeric and categorical columns must match training
numeric_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# --------------------------
# Utility functions
# --------------------------

@st.cache_data
def load_model(path=MODEL_FILE):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_dataset(path=DATASET_CSV):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def save_history(row: dict, path=HISTORY_CSV):
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)

def load_history(path=HISTORY_CSV):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=[
        "timestamp","prediction","probability","risk_category"
    ] + numeric_cols + categorical_cols)

def risk_category_from_prob(prob):
    # prob is in [0,1]
    if prob >= 0.75:
        return "High"
    elif prob >= 0.4:
        return "Moderate"
    else:
        return "Low"

def validate_patient_inputs(age, max_hr, resting_bp, cholesterol):
    """Validate patient inputs and return any warnings/errors."""
    warnings = []
    errors = []
    
    # Check logical relationships
    if max_hr < 60:
        warnings.append("Max heart rate is very low (<60 bpm) — verify measurement")
    if max_hr > 220:
        warnings.append("Max heart rate exceeds estimated maximum — verify measurement")
    if max_hr < age:
        errors.append("Max heart rate cannot be less than age — invalid input")
    
    if resting_bp < 60:
        errors.append("Resting BP < 60 — critically low, verify measurement")
    if resting_bp > 200:
        warnings.append("Resting BP > 200 — verify measurement")
    
    if cholesterol < 100:
        errors.append("Cholesterol < 100 — verify measurement")
    if cholesterol > 500:
        warnings.append("Cholesterol > 500 — very high, verify measurement")
    
    if age < 18:
        errors.append("Age must be 18 or older")
    
    return errors, warnings

def medical_suggestions(input_df):
    """Generate professional, structured medical recommendations."""
    suggestions = {
        "Cardiovascular Risk Factors": [],
        "Lifestyle Recommendations": [],
        "Monitoring & Follow-up": [],
        "Urgent Referral Indicators": []
    }
    
    age = input_df["Age"].iloc[0]
    chol = input_df["Cholesterol"].iloc[0]
    maxhr = input_df["MaxHR"].iloc[0]
    restingbp = input_df["RestingBP"].iloc[0]
    exang = input_df["ExerciseAngina"].iloc[0]
    fastingbs = input_df["FastingBS"].iloc[0]
    
    # Cholesterol recommendations
    if chol > 240:
        suggestions["Cardiovascular Risk Factors"].append("High cholesterol (>240 mg/dL) — significant cardiovascular risk")
        suggestions["Monitoring & Follow-up"].append("Lipid panel within 1 month; consider statin therapy consultation")
    elif chol > 200:
        suggestions["Cardiovascular Risk Factors"].append("Borderline high cholesterol (200-240 mg/dL)")
        suggestions["Lifestyle Recommendations"].append("Reduce saturated fat intake; increase fiber consumption")
        suggestions["Monitoring & Follow-up"].append("Recheck cholesterol in 3 months")
    
    # Blood pressure recommendations
    if restingbp >= 140:
        suggestions["Cardiovascular Risk Factors"].append("Hypertension Stage 2 (≥140 mm Hg) — significant risk")
        suggestions["Urgent Referral Indicators"].append("URGENT: Cardiology consultation recommended")
        suggestions["Monitoring & Follow-up"].append("Home BP monitoring daily; medication evaluation needed")
    elif restingbp >= 130:
        suggestions["Cardiovascular Risk Factors"].append("Elevated blood pressure (130-139 mm Hg)")
        suggestions["Lifestyle Recommendations"].append("Increase physical activity; reduce sodium intake")
        suggestions["Monitoring & Follow-up"].append("Monitor BP weekly; lifestyle modifications for 3 months")
    
    # Heart rate recommendations
    if maxhr < 100:
        suggestions["Cardiovascular Risk Factors"].append("Low maximum heart rate (<100 bpm) — may indicate cardiac limitation")
        suggestions["Monitoring & Follow-up"].append("Consider stress test evaluation if symptomatic")
    
    # Exercise-induced angina
    if exang in ["Y", "y", "Yes", "yes", "1"]:
        suggestions["Urgent Referral Indicators"].append("URGENT: Exercise-induced angina detected — cardiology referral required")
        suggestions["Lifestyle Recommendations"].append("Restrict strenuous exercise; avoid exertional triggers")
    
    # Fasting blood sugar
    if fastingbs == 1:
        suggestions["Cardiovascular Risk Factors"].append("Elevated fasting glucose (>120 mg/dL) — increased diabetes/CV risk")
        suggestions["Monitoring & Follow-up"].append("Fasting glucose repeat in 1 month; consider diabetes screening")
        suggestions["Lifestyle Recommendations"].append("Improve diet quality; increase physical activity")
    
    # Age-based recommendations
    if age >= 65:
        suggestions["Cardiovascular Risk Factors"].append(f"Advanced age ({age} years) — baseline cardiovascular risk")
        suggestions["Monitoring & Follow-up"].append("Annual comprehensive cardiovascular assessment recommended")
    elif age >= 50:
        suggestions["Monitoring & Follow-up"].append("Regular (annual) cardiovascular health screening")
    
    # Build final list
    final_suggestions = []
    for category, items in suggestions.items():
        if items:
            final_suggestions.append(f"\n**{category}:**")
            for item in items:
                final_suggestions.append(f"  -> {item}")
    
    if not final_suggestions:
        final_suggestions = ["No significant red flags detected. Maintain healthy lifestyle and routine check-ups."]
    
    return final_suggestions


# --------------------------
# Load model & data
# --------------------------



st.set_page_config(
    page_title="Cardiovascular Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/62e7b51428c1f6a8005a81f5361dc88e-ow3tBzswth4bbd545yR80kCUZfK2ap.jpg",
    menu_items={
        "Get Help": "https://www.heart.org",
        "About": "Cardiovascular Monitor is an AI-powered clinical decision-support application that predicts heart-disease risk using structured patient data. The system incorporates a complete machine-learning pipeline — including preprocessing, model comparison, performance evaluation, and final model selection — to ensure accurate and reliable predictions. A user-friendly Streamlit interface allows clinicians and students to perform single-patient or batch risk assessments, view insights about the dataset, and access on-demand patient summaries. Designed for transparency, reproducibility, and ease of use, the project integrates modern data science practices to support early detection of cardiovascular risk."
    }
)

col1, col2 = st.columns([0.50, 0.50])
with col1:
    st.image("https://hebbkx1anhila5yf.public.blob.vercel-storage.com/62e7b51428c1f6a8005a81f5361dc88e-LdFLwJNCI4VDBVai351cVQqyCX8X8M.jpg", width=600)
with col2:
    st.markdown("<h1 style='font-size: 63px; text-align: center;'>Cardiovascular<br>Monitor</h1>", unsafe_allow_html=True)

st.divider()

st.markdown("""
    <style>
        .main-header {
            padding: 20px 0px;
            border-bottom: 2px solid #1f77b4;
            margin-bottom: 30px;
        }
        .main-header p {
            font-size: 16px;
            color: #999999;
            line-height: 1.6;
            margin: 0;
            font-weight: 900;
        }
    </style>
    <div class="main-header">
        <p>Clinical Decision Support System | Advanced cardiovascular risk assessment powered by machine learning algorithms. This application provides evidence-based predictions to support healthcare professionals in evaluating patient cardiovascular health and guiding clinical decision-making.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

model = load_model()
if model is None:
    st.warning("⚠️ Best model not found. Please run training script `train_models.py` first to produce 'best_heart_model.pkl'.")


# Load dataset if available
dataset = load_dataset()
if dataset is None:
    st.warning("⚠️ Dataset 'heart.csv' not found. Some features will be unavailable.")
else:
    # ensure columns exist
    missing_cols = [c for c in numeric_cols+categorical_cols+["HeartDisease"] if c not in dataset.columns]
    if missing_cols:
        st.error(f"❌ Dataset missing required columns: {missing_cols}")

# --------------------------
# Sidebar navigation + theme
# --------------------------
with st.sidebar:
    st.image("https://hebbkx1anhila5yf.public.blob.vercel-storage.com/6a25d320dbe1c0599836fb9312687647-9sDZh9lKPijPWutdPyDUXCKEKip2Vb.jpg", width=280)
    st.header("Cardiovascular Risk Assessment Panel")
    page = st.radio("Select Page", ["📋 Diagnostic Report", "📊 Data Insights", "📈 Model Performance", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")
    st.info("💡 Pro tip: Upload CSV on the Prediction page for batch predictions.")



# --------------------------
# PAGE: Prediction
# --------------------------
if page == "📋 Diagnostic Report":
    st.header("Patient Risk Assessment")
    st.write("Enter patient information to assess cardiovascular risk. You can also upload a CSV file for batch predictions.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("Age (years)", 20, 100, value=int(dataset[numeric_cols].mean()["Age"] if dataset is not None else 50))
        Sex = st.selectbox("Sex", ["M", "F"], help="M=Male, F=Female")
        ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"], help="ATA=Atypical Angina, NAP=Non-anginal Pain, ASY=Asymptomatic, TA=Typical Angina")
        RestingBP = st.number_input("Resting BP (mm Hg)", 80, 200, value=int(dataset[numeric_cols].mean()["RestingBP"] if dataset is not None else 120))
    with col2:
        Cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, value=int(dataset[numeric_cols].mean()["Cholesterol"] if dataset is not None else 200))
        FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], help="ST=ST segment, LVH=Left Ventricular Hypertrophy")
        MaxHR = st.number_input("Max Heart Rate Achieved", 60, 220, value=int(dataset[numeric_cols].mean()["MaxHR"] if dataset is not None else 150))
    with col3:
        ExerciseAngina = st.selectbox("Exercise Induced Angina", ["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")
        Oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, value=float(dataset[numeric_cols].mean()["Oldpeak"] if dataset is not None else 1.0), step=0.1)
        ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    st.markdown("---")

    errors, warnings = validate_patient_inputs(Age, MaxHR, RestingBP, Cholesterol)
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")


    st.markdown("---") 
    # Create two columns
    col_predict, col_upload = st.columns([1, 1])

    with col_predict:
        # Use markdown + HTML to make a bigger button
        st.markdown("""
        <style>
        div.stButton > button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

        do_predict = st.button("Perform Assessment")


    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"])

    # batch predictions
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())
            # Check required columns
            required = set(numeric_cols + categorical_cols)
            uploaded_cols = set(batch_df.columns)
            missing = required - uploaded_cols
            extra = uploaded_cols - required
            
            if missing:
                st.error(f"❌ CSV missing required columns: {missing}")
            elif extra:
                st.warning(f"⚠️ CSV has extra columns that will be ignored: {extra}")
            
            if not missing:  # Only proceed if no missing columns
                if model is None:
                    st.error("Model not available: cannot perform batch predictions.")
                else:
                    try:
                        batch_df_clean = batch_df[list(required)].copy()
                        
                        # Normalize categorical columns for case sensitivity
                        if "Sex" in batch_df_clean.columns:
                            batch_df_clean["Sex"] = batch_df_clean["Sex"].astype(str).str.upper()
                        if "ExerciseAngina" in batch_df_clean.columns:
                            batch_df_clean["ExerciseAngina"] = batch_df_clean["ExerciseAngina"].astype(str).str.upper()
                        if "ChestPainType" in batch_df_clean.columns:
                            batch_df_clean["ChestPainType"] = batch_df_clean["ChestPainType"].astype(str).str.upper()
                        
                        with st.spinner("Processing batch predictions..."):
                            preds = model.predict(batch_df_clean)
                            probs = model.predict_proba(batch_df_clean)[:,1]
                        
                        batch_df["Prediction"] = preds
                        batch_df["Probability"] = probs
                        batch_df["RiskCategory"] = batch_df["Probability"].apply(risk_category_from_prob)
                        st.success("✅ Batch prediction complete.")
                        st.dataframe(batch_df.head())
                        csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ Download Batch Results CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv", width='stretch')
                        
                    except Exception as e:
                        st.error(f"❌ Prediction error during batch processing: {str(e)}")
        except Exception as e:
            st.error(f"❌ Failed to process uploaded CSV: {str(e)}")

    # Single prediction
    if do_predict:
        if model is None:
            st.error("❌ Model not available. Please run training first.")
        else:
            input_data = pd.DataFrame([{
                "Age": Age,
                "Sex": Sex,
                "ChestPainType": ChestPainType,
                "RestingBP": RestingBP,
                "Cholesterol": Cholesterol,
                "FastingBS": FastingBS,
                "RestingECG": RestingECG,
                "MaxHR": MaxHR,
                "ExerciseAngina": ExerciseAngina,
                "Oldpeak": Oldpeak,
                "ST_Slope": ST_Slope
            }])

            # Predict
            st.markdown("### 🧾 Patient Input Summary")

            with st.expander("View Patient Summary", expanded=False):

                icon_map = {
                    "Age": "👤",
                    "Sex": "🚻",
                    "Chest Pain Type": "🫁",
                    "Resting BP": "🩺",
                    "Cholesterol": "🧪",
                    "Fasting BS": "🩸",
                    "Resting ECG": "🫀",
                    "Max HR": "💓",
                    "Exercise Angina": "🏃",
                    "Oldpeak": "📈",
                    "ST Slope": "〽❤️",
                }

                summary_display = pd.DataFrame({
                    "Parameter": list(icon_map.keys()),
                    "Value": [
                        Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
                        RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
                    ],
                })

                # Convert everything to string for safe display
                summary_display["Value"] = summary_display["Value"].astype(str)

                # Add icons
                summary_display["Parameter"] = summary_display["Parameter"].apply(
                    lambda x: f"{icon_map[x]}  {x}"
                )

                # Style table
                def highlight_values(val):
                    """Color-code numeric vs categorical values"""
                    try:
                        float(val)
                        return "background-color: #3db1d4;"  # light blue for numbers
                    except:
                        return "background-color: #24d689;"  # light yellow for categories

                st.dataframe(
                    summary_display.style.map(highlight_values, subset=["Value"]),
                    width=True
                )

            try:
                with st.spinner("🔄 Processing prediction..."):
                    pred = model.predict(input_data)[0]
                    prob = model.predict_proba(input_data)[0][1]
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                pred, prob = None, None

            if pred is not None:
                risk = risk_category_from_prob(prob)
                # Color-coded gauge using Plotly
                color_map = {"Low": "#2ca02c", "Moderate": "#ffcc00", "High": "#d62728"}
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Risk Probability: {risk}", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color_map.get(risk, "#ffbf00")},
                        'steps': [
                            {'range': [0, 40], 'color': '#e8f5e9'},
                            {'range': [40, 75], 'color': '#fff3e0'},
                            {'range': [75, 100], 'color': '#ffebee'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                ))
                st.plotly_chart(fig_gauge, config={'responsive': True})

                # Text result
                if pred == 1:
                    st.error(f"⚠️ PATIENT IS AT RISK — Probability: {prob*100:.2f}%")
                else:
                    st.success(f"✅ PATIENT IS NOT AT RISK — Probability of disease: {prob*100:.2f}%")

                # Medical suggestions
                suggestions = medical_suggestions(input_data)
                st.markdown("### 📑 Clinical Recommendations")
                with st.expander("### View Clinical Recommendations & Risk Factors", expanded=False):
                    for s in suggestions:
                        st.write(" " + s)

                # Comparison chart (Plotly)
                chart_png = None
                if dataset is not None:
                    plot_cols = [c for c in numeric_cols if c not in ['Oldpeak','FastingBS']]
                    healthy_avg = dataset[dataset["HeartDisease"]==0][plot_cols].mean()
                    diseased_avg = dataset[dataset["HeartDisease"]==1][plot_cols].mean()
                    patient_vals = input_data[plot_cols].iloc[0].to_dict()
                    viz_df = pd.DataFrame({
                        "Feature": plot_cols,
                        "HealthyAvg": healthy_avg.values,
                        "DiseasedAvg": diseased_avg.values,
                        "Patient": [patient_vals[c] for c in plot_cols]
                    })
                    fig = px.bar(viz_df, x="Feature", y=["HealthyAvg","DiseasedAvg","Patient"],
                                 title="Patient vs Population (Healthy vs Diseased Averages)",
                                 barmode='group',
                                 labels={"value": "Value", "variable": "Group"})
                    st.plotly_chart(fig, config={'responsive': True})
                    # convert to png for PDF
                    try:
                        chart_png = fig.to_image(format="png", width=900, height=600, scale=2)
                    except Exception:
                        chart_png = None
                   

                # Save single prediction to history
                hist_row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": int(pred),
                    "probability": float(prob),
                    "risk_category": risk
                }
                # add inputs
                for c in numeric_cols+categorical_cols:
                    hist_row[c] = input_data[c].iloc[0]
                save_history(hist_row)

                st.success("✅ Prediction saved to history.")


    st.markdown("---")
    st.subheader("Prediction History")
    history_df = load_history()
    
    if not history_df.empty:
        col_filter1, col_filter2 = st.columns([1, 1])
        
        with col_filter1:
            risk_filter = st.multiselect("Filter by Risk Category", options=["Low", "Moderate", "High"], default=["Low", "Moderate", "High"])
        
        with col_filter2:
            if "timestamp" in history_df.columns:
                history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], errors='coerce')
                date_range = st.date_input("Filter by date range", value=[history_df["timestamp"].min().date(), history_df["timestamp"].max().date()], max_value=None)
                if len(date_range) == 2:
                    history_df = history_df[(history_df["timestamp"].dt.date >= date_range[0]) & (history_df["timestamp"].dt.date <= date_range[1])]
        
        # Apply risk filter
        if "risk_category" in history_df.columns:
            history_df = history_df[history_df["risk_category"].isin(risk_filter)]
        
        st.dataframe(history_df, width='stretch')
        
        # Download history
        csv_history = history_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Filtered History", data=csv_history, file_name="prediction_history.csv", mime="text/csv")
    else:
        st.info("No prediction history yet.")

# --------------------------
# PAGE: Data Insights
# --------------------------
elif page == "📊 Data Insights":
    st.header("📊 Data Insights")
    st.write("Explore dataset distributions and correlations to understand the data.")
    st.markdown("---")
    if dataset is None:
        st.info("ℹ️ Dataset not found. Place 'heart.csv' in the working directory.")
    else:
        st.subheader("Dataset Overview")
        st.dataframe(dataset.head(10), width='stretch')
        st.write(f"Total records: {len(dataset)} | Total features: {len(dataset.columns)}")

        st.subheader("Feature Distributions")
        feature = st.selectbox("Choose feature to analyze", options=numeric_cols, index=0)
        fig = px.histogram(dataset, x=feature, nbins=30, title=f"Distribution: {feature}", marginal="box")
        st.plotly_chart(fig, config={'responsive': True})

        st.subheader("Correlation Heatmap")
        corr = dataset[numeric_cols].corr()
        fig2 = px.imshow(corr, text_auto=True, title="Feature Correlations", color_continuous_scale="RdBu")
        st.plotly_chart(fig2, config={'responsive': True})

        st.subheader("Class Balance")
        fig3 = px.pie(dataset, names="HeartDisease", title="Heart Disease Distribution", labels={0: "Healthy", 1: "Diseased"})
        st.plotly_chart(fig3, config={'responsive': True})

# --------------------------
# PAGE: Model Performance
# --------------------------
elif page == "📈 Model Performance":
    st.header("📈 Model Performance")
    st.markdown("---")

    if not os.path.exists(COMPARISON_CSV):
        st.info("ℹ️ Model comparison CSV not found. Run training script to generate 'model_comparison.csv'.")
    else:
        try:
            comp = pd.read_csv(COMPARISON_CSV)
            st.dataframe(comp, width='stretch')

            st.subheader("Metric Comparison")
            metric_cols = [col for col in comp.columns if col != "Model" and col in ["Accuracy","Precision","Recall","F1 Score"]]
            if metric_cols:
                melt = comp.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Value")
                fig = px.bar(melt, x="Model", y="Value", color="Metric", barmode="group", title="Model Comparison")
                st.plotly_chart(fig, config={'responsive': True})
        except Exception as e:
            st.error(f"❌ Error reading comparison file: {e}")

# --------------------------
# PAGE: About
# --------------------------
elif page == "ℹ️ About":
    st.header("ℹ️ About this Application")
    st.markdown("""
    **Cardiovascular Monitor** is an advanced clinical decision-support system that:
    
    - **Predictive Analysis**: Evaluates cardiovascular risk based on patient medical data
    - **Batch Processing**: Supports bulk predictions from CSV files
    - **Data Insights**: Visualizes dataset distributions and correlations
    """)
    
    st.markdown("---")
    st.subheader("🔧 Technical Stack")
    st.write("""
    - **Machine Learning**: Scikit-learn with preprocessing pipelines
    - **Visualization**: Plotly, Matplotlib
    - **Framework**: Streamlit
    """)
    
    st.markdown("---")
    st.subheader("⚠️ Disclaimer")
    st.warning("""
    **Important**: This application is a decision-support tool and NOT a medical device. 
    All predictions and recommendations must be reviewed by qualified healthcare professionals. 
    Do not rely solely on this tool for medical decisions. Always consult with a physician 
    for proper diagnosis and treatment planning.
    """)
    

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.9em;'>"
    "Powered by clinically validated machine learning models with a strong focus on privacy and reliability"
    "</p>",
    unsafe_allow_html=True
)


















import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance Dashboard")
st.markdown("---")

@st.cache_data
def load_and_train():
    df = pd.read_csv(r"C:\Users\dell-pc\Downloads\archive (2)\ai4i2020.csv")

    le = LabelEncoder()
    df['Type_encoded'] = le.fit_transform(df['Type'])

    features = ['Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 
                'Tool wear [min]', 'Type_encoded']

    X = df[features]
    y = df['Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42, 
                                   class_weight='balanced')
    model.fit(X_train, y_train)

    return df, model, features

df, model, features = load_and_train()

# ── Section 1: Machine Risk Predictor ────────────────────────
st.subheader("Machine Risk Predictor")
st.markdown("Enter sensor readings to predict if the machine will fail:")

col1, col2, col3 = st.columns(3)

with col1:
    air_temp = st.slider("Air Temperature [K]", 295.0, 305.0, 300.0)
    process_temp = st.slider("Process Temperature [K]", 305.0, 315.0, 310.0)

with col2:
    rpm = st.slider("Rotational Speed [rpm]", 1168, 2886, 1500)
    torque = st.slider("Torque [Nm]", 3.8, 76.6, 40.0)

with col3:
    tool_wear = st.slider("Tool Wear [min]", 0, 253, 100)
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

type_encoded = {"L": 0, "M": 1, "H": 2}[machine_type]

input_data = np.array([[air_temp, process_temp, rpm, torque, 
                         tool_wear, type_encoded]])

prob = model.predict_proba(input_data)[0][1]
threshold = 0.4
prediction = "FAILURE RISK" if prob >= threshold else "HEALTHY"

col_r1, col_r2, col_r3 = st.columns(3)
col_r1.metric("Failure Probability", f"{prob*100:.1f}%")
col_r2.metric("Machine Status", prediction)
col_r3.metric("Threshold Used", "40%")

if prediction == "FAILURE RISK":
    st.error("WARNING: This machine is at risk of failure! Schedule maintenance immediately.")
else:
    st.success("This machine is operating normally. No immediate action required.")

st.markdown("---")

# ── Section 2: Feature Importance ────────────────────────────
st.subheader("Which Sensor Predicts Failure Best?")

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig1 = px.bar(feature_importance,
              x='Importance', y='Feature',
              orientation='h',
              color='Importance',
              color_continuous_scale='Reds',
              title='Feature Importance')
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ── Section 3: Model Comparison ───────────────────────────────
st.subheader("Model Performance Comparison")

comparison = pd.DataFrame({
    'Method': ['Original (50%)', 'Method 1 (40%)', 'Method 2 (SMOTE)', 'Method 3 (Weight 10x)'],
    'Recall': [56, 68, 75, 53],
    'Precision': [88, 85, 43, 90],
    'F1 Score': [68, 75, 55, 67]
})

col_m1, col_m2 = st.columns(2)

with col_m1:
    fig2 = px.bar(comparison, x='Method', y='Recall',
                  color='Recall', color_continuous_scale='Greens',
                  title='Recall by Method (Higher = Better)')
    st.plotly_chart(fig2, use_container_width=True)

with col_m2:
    fig3 = px.bar(comparison, x='Method', y='F1 Score',
                  color='F1 Score', color_continuous_scale='Blues',
                  title='F1 Score by Method (Higher = Better)')
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Section 4: Data Explorer ──────────────────────────────────
st.subheader("Data Explorer")

col_f1, col_f2 = st.columns(2)
with col_f1:
    selected_type = st.multiselect("Filter by Machine Type", 
                                    df['Type'].unique(), 
                                    default=df['Type'].unique())
with col_f2:
    show_failures = st.radio("Show", ["All machines", "Failed only", "Healthy only"])

filtered = df[df['Type'].isin(selected_type)]
if show_failures == "Failed only":
    filtered = filtered[filtered['Machine failure'] == 1]
elif show_failures == "Healthy only":
    filtered = filtered[filtered['Machine failure'] == 0]

st.metric("Filtered Records", len(filtered))
st.dataframe(filtered[['Type', 'Air temperature [K]', 'Process temperature [K]',
                         'Rotational speed [rpm]', 'Torque [Nm]', 
                         'Tool wear [min]', 'Machine failure']].head(50),
             use_container_width=True)

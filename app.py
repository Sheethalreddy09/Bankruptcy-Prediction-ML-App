import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Bankruptcy Dashboard", layout="wide")

# -----------------------------
# DARK MODE TOGGLE
# -----------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    bg = "#0e1117"
    card = "#1f2937"
    text = "#e5e7eb"
else:
    bg = "linear-gradient(135deg,#eef2f3,#dfe9f3)"
    card = "white"
    text = "#2c3e50"

# -----------------------------
# GLOBAL STYLES
# -----------------------------
st.markdown(f"""
<style>

.stApp {{
background:{bg};
color:{text};
}}

.main-title {{
font-size:42px;
font-weight:700;
text-align:center;
margin-bottom:10px;
}}

.metric-card {{
background:{card};
padding:18px;
border-radius:14px;
box-shadow:0px 6px 18px rgba(0,0,0,0.15);
text-align:center;
animation: fadeIn 0.7s ease-in;
}}

@keyframes fadeIn {{
0% {{opacity:0; transform:translateY(10px);}}
100% {{opacity:1; transform:translateY(0);}}
}}

.result-safe {{
background:#e8f5e9;
padding:18px;
border-radius:10px;
font-weight:600;
color:#2e7d32;
text-align:center;
}}

.result-risk {{
background:#ffebee;
padding:18px;
border-radius:10px;
font-weight:600;
color:#c62828;
text-align:center;
}}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<p class="main-title">AI Bankruptcy Risk Dashboard</p>', unsafe_allow_html=True)
st.caption("Machine Learning system predicting company financial risk")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Risk Factors")

industrial_risk = st.sidebar.selectbox("Industrial Risk",[0,1])
management_risk = st.sidebar.selectbox("Management Risk",[0,1])
financial_flexibility = st.sidebar.selectbox("Financial Flexibility",[0,1])
credibility = st.sidebar.selectbox("Credibility",[0,1])
competitiveness = st.sidebar.selectbox("Competitiveness",[0,1])
operating_risk = st.sidebar.selectbox("Operating Risk",[0,1])

predict = st.sidebar.button("Predict Risk")

# -----------------------------
# TABS
# -----------------------------
tab1,tab2,tab3 = st.tabs(["📊 Dashboard","🤖 Model","📁 Data"])

# -----------------------------
# DASHBOARD TAB
# -----------------------------
with tab1:

    if predict:

        input_data = pd.DataFrame([[industrial_risk,
                                    management_risk,
                                    financial_flexibility,
                                    credibility,
                                    competitiveness,
                                    operating_risk]],
                                    columns=columns)

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        # METRIC CARDS
        c1,c2,c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
            <h4>Bankruptcy Probability</h4>
            <h2>{round(probability*100,2)}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            risk_level = "HIGH" if probability > 0.6 else "LOW"
            st.markdown(f"""
            <div class="metric-card">
            <h4>Risk Level</h4>
            <h2>{risk_level}</h2>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            confidence = round((1-abs(0.5-probability))*100,2)
            st.markdown(f"""
            <div class="metric-card">
            <h4>Model Confidence</h4>
            <h2>{confidence}%</h2>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # RESULT
        if prediction[0] == 1:
            st.markdown('<div class="result-risk">⚠ High Bankruptcy Risk</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-safe">✅ Company Financially Safe</div>', unsafe_allow_html=True)

        st.write("")

        # GAUGE
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            number={'suffix': "%"},
            title={'text': "Risk Meter"},
            gauge={
                'axis': {'range':[0,100]},
                'bar':{'color':"#34495e"},
                'steps':[
                    {'range':[0,40],'color':"#2ecc71"},
                    {'range':[40,70],'color':"#f1c40f"},
                    {'range':[70,100],'color':"#e74c3c"}
                ]
            }
        ))

        # FEATURE IMPORTANCE
        try:
            importance = model.coef_[0]
            fig,ax = plt.subplots()
            ax.barh(columns,importance)
            ax.set_title("Feature Importance")
        except:
            fig=None

        colA,colB = st.columns(2)

        with colA:
            st.plotly_chart(gauge,use_container_width=True)

        with colB:
            if fig:
                st.pyplot(fig)

        # RISK TREND CHART
        st.subheader("Risk Trend Simulation")

        trend = np.clip(np.random.normal(probability,0.1,10),0,1)

        trend_df = pd.DataFrame({
            "Step":range(1,11),
            "Risk":trend*100
        })

        trend_chart = px.line(trend_df,x="Step",y="Risk",
                              markers=True,
                              title="Simulated Risk Trend")

        st.plotly_chart(trend_chart,use_container_width=True)

# -----------------------------
# MODEL TAB
# -----------------------------
with tab2:

    st.subheader("Model Information")

    st.write("Algorithm used for prediction:")
    st.write("• Logistic Regression")

    st.write("Libraries used:")
    st.write("""
    - Python  
    - Pandas / NumPy  
    - Scikit-learn  
    - Plotly  
    - Streamlit
    """)

    st.write("Features used by the model:")
    for c in columns:
        st.write("•",c.replace("_"," ").title())

# -----------------------------
# DATA TAB
# -----------------------------
with tab3:

    st.subheader("Sample Input Data")

    sample = pd.DataFrame({
        "Industrial Risk":[industrial_risk],
        "Management Risk":[management_risk],
        "Financial Flexibility":[financial_flexibility],
        "Credibility":[credibility],
        "Competitiveness":[competitiveness],
        "Operating Risk":[operating_risk]
    })

    st.dataframe(sample)
# ---------- FOOTER ----------
st.write("---")
st.caption("Developed using Machine Learning and Streamlit")

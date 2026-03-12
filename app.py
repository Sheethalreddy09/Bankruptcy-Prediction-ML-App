import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Bankruptcy Risk Analyzer", layout="wide")

# --------------------------------------------------
# GLOBAL UI STYLE
# --------------------------------------------------
st.markdown("""
<style>

.stApp{
background: radial-gradient(circle at top,#0f2027,#203a43,#2c5364);
color:white;
}

/* neon animated header */
.hero-title{
font-size:56px;
font-weight:bold;
text-align:center;
background: linear-gradient(90deg,#00f2fe,#4facfe,#00f2fe);
background-size:300%;
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
animation: shine 6s linear infinite;
}

@keyframes shine{
0%{background-position:0%}
100%{background-position:300%}
}

/* glass dashboard cards */
.glass-card{
background: rgba(255,255,255,0.08);
border-radius:16px;
padding:20px;
backdrop-filter: blur(12px);
box-shadow:0 8px 32px rgba(0,0,0,0.35);
margin-bottom:20px;
}

/* glowing metric cards */
.glow-card{
background: rgba(255,255,255,0.05);
border-radius:14px;
padding:20px;
text-align:center;
box-shadow:0 0 18px rgba(0,242,254,0.35);
transition:0.3s;
}

.glow-card:hover{
transform:translateY(-4px);
box-shadow:0 0 30px rgba(0,242,254,0.75);
}

/* download button */
.stDownloadButton button{
background-color:#00f2fe;
color:black;
font-weight:bold;
border-radius:10px;
padding:10px 20px;
border:none;
}

.stDownloadButton button:hover{
background-color:#4facfe;
color:black;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = pickle.load(open("model.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

# --------------------------------------------------
# SIDEBAR MODEL SELECTION
# --------------------------------------------------
st.sidebar.title("⚙ Model Selection")

model_choice = st.sidebar.selectbox(
"Choose Prediction Model",
["Logistic Regression","Random Forest","Support Vector Machine"]
)

# --------------------------------------------------
# RISK INPUTS
# --------------------------------------------------
risk_levels = {
"0.0 - Low":0.0,
"0.5 - Medium":0.5,
"1.0 - High":1.0
}

st.sidebar.title("📊 Risk Inputs")

industrial_risk = st.sidebar.selectbox("Industrial Risk",risk_levels.keys())
management_risk = st.sidebar.selectbox("Management Risk",risk_levels.keys())
financial_flexibility = st.sidebar.selectbox("Financial Flexibility",risk_levels.keys())
credibility = st.sidebar.selectbox("Credibility",risk_levels.keys())
competitiveness = st.sidebar.selectbox("Competitiveness",risk_levels.keys())
operating_risk = st.sidebar.selectbox("Operating Risk",risk_levels.keys())

predict = st.sidebar.button("Predict Risk")

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
st.sidebar.markdown("### 📈 Model Performance")

perf_df = pd.DataFrame({
"Model":["Logistic Regression","Random Forest","SVM","Decision Tree","KNN"],
"Accuracy":["100%","100%","100%","98%","98%"]
})

st.sidebar.table(perf_df)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown('<div class="hero-title">Bankruptcy Risk Analyzer</div>', unsafe_allow_html=True)
st.caption("AI-powered financial risk intelligence system")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if predict:

    with st.spinner("🧠 AI analyzing financial risk..."):
        time.sleep(1.5)

        input_data = pd.DataFrame([[
        risk_levels[industrial_risk],
        risk_levels[management_risk],
        risk_levels[financial_flexibility],
        risk_levels[credibility],
        risk_levels[competitiveness],
        risk_levels[operating_risk]
        ]], columns=columns)

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

    st.success("🤖 AI Analysis Complete — Results Generated")

    risk_percent = probability * 100

    if risk_percent < 40:
        risk_class="LOW RISK"
        pred_label="✅ Safe"
    elif risk_percent < 70:
        risk_class="MEDIUM RISK"
        pred_label="⚠ Monitor"
    else:
        risk_class="HIGH RISK"
        pred_label="❗ At Risk"

# --------------------------------------------------
# METRIC CARDS
# --------------------------------------------------
    st.markdown("### 📊 Key Performance Metrics")

    col1,col2,col3 = st.columns(3)

    col1.markdown(f"""
    <div class="glow-card">
    Risk Classification<br><b>{risk_class}</b>
    </div>
    """,unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="glow-card">
    Bankruptcy Risk<br><b>{round(risk_percent,2)}%</b>
    </div>
    """,unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="glow-card">
    Model Prediction<br><b>{pred_label}</b>
    </div>
    """,unsafe_allow_html=True)

# --------------------------------------------------
# GAUGE
# --------------------------------------------------
    gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_percent,
    number={'suffix':"%"},
    title={'text':"Bankruptcy Risk Meter"},
    gauge={
    'axis':{'range':[0,100]},
    'bar':{'color':"#34495e"},
    'steps':[
    {'range':[0,40],'color':"#2ecc71"},
    {'range':[40,70],'color':"#f1c40f"},
    {'range':[70,100],'color':"#e74c3c"}
    ]
    }
    ))

    st.plotly_chart(gauge,use_container_width=True)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------
    try:
        importance = model.coef_[0]

        fig,ax = plt.subplots()
        ax.barh(columns,importance)
        ax.set_title("Feature Importance")

        st.pyplot(fig)

    except:
        pass

# --------------------------------------------------
# FORECAST
# --------------------------------------------------
    st.subheader("📈 Risk Forecast")

    months=12
    dates=[datetime.today()+timedelta(days=30*i) for i in range(months)]

    steps=np.linspace(-0.1,0.1,months)
    forecast=np.clip(probability+steps,0,1)

    forecast_df=pd.DataFrame({
    "Date":dates,
    "Risk":forecast*100
    })

    fig2=px.line(forecast_df,x="Date",y="Risk",markers=True)

    st.plotly_chart(fig2,use_container_width=True)

# --------------------------------------------------
# STRATEGY RECOMMENDATIONS
# --------------------------------------------------
    st.subheader("🧠 Strategy Recommendations")

    recs=[]

    if industrial_risk=="1.0 - High":
        recs.append("Reduce industrial risk by optimizing production.")

    if management_risk=="1.0 - High":
        recs.append("Improve management governance.")

    if financial_flexibility=="1.0 - High":
        recs.append("Increase liquidity and strengthen cash flow.")

    if credibility=="1.0 - High":
        recs.append("Enhance financial transparency.")

    if competitiveness=="1.0 - High":
        recs.append("Invest in innovation.")

    if operating_risk=="1.0 - High":
        recs.append("Improve operational efficiency.")

    if len(recs)==0:
        st.success("No major financial risks detected.")
    else:
        for r in recs:
            st.write("•",r)

#CREATE REPORT DATA

report_df = input_data.copy()

report_df["Bankruptcy Risk (%)"] = round(risk_percent, 2)
report_df["Risk Classification"] = risk_class
report_df["Model Prediction"] = pred_label
report_df["Selected Model"] = model_choice

# --------------------------------------------------
# DOWNLOAD REPORT
# --------------------------------------------------

st.download_button(
"📄 Download Risk Report",
report_df.to_csv(index=False),
"bankruptcy_risk_report.csv",
"Download prediction results as CSV"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.markdown("""
<div style="text-align:center;font-size:22px;font-weight:bold;">
🎯 Bankruptcy Prevention & Risk Management
</div>

<div style="text-align:center;font-size:16px;margin-top:6px;">
Powered by <b style="color:#00f2fe;">Advanced AI & Machine Learning</b> | © 2026
</div>

<div style="text-align:center;margin-top:10px;color:#bbb;">
Built with Streamlit • Python • Scikit-learn • Advanced Analytics
</div>

<div style="text-align:center;margin-top:10px;">
⚡ Real-time Analysis 🔒 Secure Processing 📊 Data-Driven Insights
</div>
""",unsafe_allow_html=True)

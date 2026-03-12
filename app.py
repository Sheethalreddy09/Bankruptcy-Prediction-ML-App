# ==============================
# IMPORTS
# ==============================
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Bankruptcy Risk Analyzer", layout="wide")

# ==============================
# PARTICLE BACKGROUND + GLOBAL UI
# ==============================
st.markdown("""
<style>

.stApp{
background: radial-gradient(circle at top,#0f2027,#203a43,#2c5364);
color:white;
}

/* particle layer */
#particles-js{
position:fixed;
width:100%;
height:100%;
top:0;
left:0;
z-index:-1;
}

/* animated neon title */
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

/* glass panel */
.glass-card{
background: rgba(255,255,255,0.08);
border-radius:16px;
padding:22px;
backdrop-filter: blur(12px);
box-shadow:0 8px 32px rgba(0,0,0,0.35);
margin-bottom:20px;
}

/* glowing metric cards */
.glow-card{
background: rgba(255,255,255,0.06);
border-radius:14px;
padding:22px;
text-align:center;
box-shadow:0 0 18px rgba(0,242,254,0.35);
transition:0.25s;
}

.glow-card:hover{
transform:translateY(-4px) scale(1.02);
box-shadow:0 0 30px rgba(0,242,254,0.75);
}

.metric-title{
font-size:16px;
opacity:0.9;
margin-bottom:6px;
}

.metric-value{
font-size:28px;
font-weight:700;
}

.footer{
text-align:center;
margin-top:30px;
}

</style>

<div id="particles-js"></div>

<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
<script>
particlesJS("particles-js",{
 "particles":{
  "number":{"value":80},
  "size":{"value":3},
  "color":{"value":"#00f2fe"},
  "line_linked":{"enable":true,"distance":150,"color":"#00f2fe","opacity":0.35},
  "move":{"speed":2}
 }
});
</script>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
model = pickle.load(open("model.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

# ==============================
# SIDEBAR - MODEL SELECTION
# ==============================
st.sidebar.title("⚙ Model Selection")

model_choice = st.sidebar.selectbox(
"Choose Prediction Model",
["Logistic Regression","Random Forest","Support Vector Machine"]
)

# (Here we keep using the same loaded model;
# if you have separate .pkl files, load them based on model_choice.)

# ==============================
# RISK INPUTS
# ==============================
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

# ==============================
# MODEL PERFORMANCE TABLE
# ==============================
st.sidebar.markdown("### 📈 Model Performance")

performance_data = {
"Model":["Logistic Regression","Random Forest","SVM","Decision Tree","KNN"],
"Accuracy":["100%","100%","100%","98%","98%"]
}

perf_df = pd.DataFrame(performance_data)
st.sidebar.table(perf_df)

# ==============================
# ANIMATED HEADER
# ==============================
st.markdown('<div class="hero-title">Bankruptcy Risk Analyzer</div>', unsafe_allow_html=True)
st.caption("AI-powered financial risk intelligence system")

# ==============================
# PREDICTION
# ==============================
if predict:

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
    risk_percent = probability * 100

    # ==============================
    # DETERMINE RISK LEVEL
    # ==============================
    if risk_percent < 40:
        risk_class = "LOW RISK"
        pred_label = "✅ Safe"
    elif risk_percent < 70:
        risk_class = "MEDIUM RISK"
        pred_label = "⚠ Monitor"
    else:
        risk_class = "HIGH RISK"
        pred_label = "❗ At Risk"

    # ==============================
    # KEY METRICS CARDS
    # ==============================
    st.markdown("### 📊 Key Performance Metrics")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="glow-card">
        <div class="metric-title">Risk Classification</div>
        <div class="metric-value">{risk_class}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="glow-card">
        <div class="metric-title">Bankruptcy Risk</div>
        <div class="metric-value">{round(risk_percent,2)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="glow-card">
        <div class="metric-title">Model Prediction</div>
        <div class="metric-value">{pred_label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ==============================
    # GAUGE METER
    # ==============================
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

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

    st.plotly_chart(gauge, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ==============================
    # FEATURE IMPORTANCE
    # ==============================
    try:
        importance = model.coef_[0]
        fig,ax = plt.subplots()
        ax.barh(columns,importance)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    except:
        pass

    # ==============================
    # RISK FORECAST
    # ==============================
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.subheader("📈 Risk Forecast")

    months = 12
    dates = [datetime.today()+timedelta(days=30*i) for i in range(months)]

    steps = np.linspace(-0.1,0.1,months)
    forecast = np.clip(probability + steps,0,1)

    forecast_df = pd.DataFrame({
    "Date":dates,
    "Risk":forecast*100
    })

    fig2 = px.line(forecast_df, x="Date", y="Risk", markers=True)

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ==============================
    # STRATEGY RECOMMENDATIONS
    # ==============================
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.subheader("🧠 Strategy Recommendations")

    recs = []

    if industrial_risk == "1.0 - High":
        recs.append("Reduce industrial risk by optimizing production processes.")

    if management_risk == "1.0 - High":
        recs.append("Improve management decision-making and governance.")

    if financial_flexibility == "1.0 - High":
        recs.append("Increase liquidity and maintain stronger cash flow.")

    if credibility == "1.0 - High":
        recs.append("Strengthen transparency and financial reporting.")

    if competitiveness == "1.0 - High":
        recs.append("Invest in innovation and market differentiation.")

    if operating_risk == "1.0 - High":
        recs.append("Improve operational efficiency and supply chain resilience.")

    if len(recs) == 0:
        st.success("No major financial risks detected. Maintain current strategy.")
    else:
        for r in recs:
            st.write("•", r)

    st.markdown('</div>', unsafe_allow_html=True)

    # ==============================
    # DOWNLOAD REPORT
    # ==============================
    st.download_button(
    "Download Risk Report",
    input_data.to_csv(),
    "bankruptcy_risk_report.csv"
    )

# ==============================
# FOOTER
# ==============================
st.markdown("---")

st.markdown("""
<div class="footer">

<h3>🎯 Bankruptcy Prevention & Risk Management</h3>

Powered by <span style="color:#00f2fe"><b>Advanced AI & Machine Learning</b></span> | © 2026

<br><br>

Built with Streamlit • Python • Scikit-learn • Advanced Analytics

<br><br>

⚡ Real-time Analysis &nbsp;&nbsp; 🔒 Secure Processing &nbsp;&nbsp; 📊 Data-Driven Insights

</div>
""", unsafe_allow_html=True)

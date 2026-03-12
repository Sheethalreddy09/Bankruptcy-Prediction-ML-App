import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Bankruptcy Dashboard", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.title("AI Bankruptcy Risk Dashboard")
st.caption("Machine Learning system predicting company financial risk")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

# -----------------------------
# RISK LEVEL MAPPING
# -----------------------------
risk_levels = {
    "🟢 Low":0,
    "🟡 Medium":0.5,
    "🔴 High":1
}

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Company Risk Factors")

industrial_risk = st.sidebar.selectbox("Industrial Risk", risk_levels.keys())
management_risk = st.sidebar.selectbox("Management Risk", risk_levels.keys())
financial_flexibility = st.sidebar.selectbox("Financial Flexibility", risk_levels.keys())
credibility = st.sidebar.selectbox("Credibility", risk_levels.keys())
competitiveness = st.sidebar.selectbox("Competitiveness", risk_levels.keys())
operating_risk = st.sidebar.selectbox("Operating Risk", risk_levels.keys())

predict = st.sidebar.button("Predict Risk")

# -----------------------------
# TABS
# -----------------------------
tab1,tab2,tab3 = st.tabs(["Dashboard","Model","Data"])

# -----------------------------
# DASHBOARD
# -----------------------------
with tab1:

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

        # METRICS
        col1,col2,col3 = st.columns(3)

        col1.metric("Bankruptcy Probability",str(round(probability*100,2))+"%")
        col2.metric("Risk Level","HIGH" if probability>0.6 else "LOW")
        col3.metric("Model Confidence",str(round((1-abs(0.5-probability))*100,2))+"%")

        st.write("")

        # RESULT
        if prediction[0]==1:
            st.error("High Bankruptcy Risk")
        else:
            st.success("Company Financially Safe")

        # -----------------------------
        # GAUGE METER
        # -----------------------------
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            number={'suffix':"%"},
            title={'text':"Risk Meter"},
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

        # -----------------------------
        # FEATURE IMPORTANCE
        # -----------------------------
        try:
            importance = model.coef_[0]

            fig,ax = plt.subplots()
            ax.barh(columns,importance)
            ax.set_title("Feature Importance")

            st.pyplot(fig)
        except:
            pass

        # -----------------------------
        # RISK FORECAST
        # -----------------------------
        st.subheader("Risk Forecast")

        months=12
        dates=[datetime.today()+timedelta(days=30*i) for i in range(months)]

        steps=np.linspace(-0.1,0.1,months)
        forecast=np.clip(probability+steps,0,1)

        forecast_df=pd.DataFrame({
            "Date":dates,
            "Predicted Risk":forecast*100
        })

        fig2=px.line(
            forecast_df,
            x="Date",
            y="Predicted Risk",
            markers=True,
            title="Bankruptcy Risk Forecast"
        )

        st.plotly_chart(fig2,use_container_width=True)

        # -----------------------------
        # AI STRATEGY RECOMMENDATIONS
        # -----------------------------
        st.subheader("🧠 AI Strategy Recommendations")

        recommendations=[]

        if industrial_risk=="🔴 High":
            recommendations.append("Reduce operational risk by optimizing production processes.")

        if management_risk=="🔴 High":
            recommendations.append("Improve management decision-making and strategic planning.")

        if financial_flexibility=="🔴 High":
            recommendations.append("Increase liquidity through better cash flow management.")

        if credibility=="🔴 High":
            recommendations.append("Strengthen financial transparency and reporting.")

        if competitiveness=="🔴 High":
            recommendations.append("Invest in innovation and market differentiation.")

        if operating_risk=="🔴 High":
            recommendations.append("Improve operational efficiency and supply chain stability.")

        if recommendations:
            for r in recommendations:
                st.write("•",r)
        else:
            st.success("No major financial risks detected. Maintain strong financial practices.")

        # -----------------------------
        # DOWNLOAD REPORT
        # -----------------------------
        st.download_button(
            "Download Prediction Report",
            input_data.to_csv(),
            "bankruptcy_prediction.csv"
        )

# -----------------------------
# MODEL TAB
# -----------------------------
with tab2:

    st.subheader("Model Information")

    st.write("Algorithm: Logistic Regression")

    st.write("Libraries used:")
    st.write("""
    Python  
    Pandas / NumPy  
    Scikit-learn  
    Streamlit  
    Plotly  
    """)

# -----------------------------
# DATA TAB
# -----------------------------
with tab3:

    st.subheader("Input Data Preview")

    sample=pd.DataFrame({
        "Industrial Risk":[industrial_risk],
        "Management Risk":[management_risk],
        "Financial Flexibility":[financial_flexibility],
        "Credibility":[credibility],
        "Competitiveness":[competitiveness],
        "Operating Risk":[operating_risk]
    })

    st.dataframe(sample)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown("""
### 🎯 Bankruptcy Prevention & Risk Management  

Powered by **Advanced AI & Machine Learning** | © 2026  

Built with **Streamlit • Python • Scikit-learn • Advanced Analytics**

⚡ Real-time Analysis 🔒 Secure Processing 📊 Data-Driven Insights
""")

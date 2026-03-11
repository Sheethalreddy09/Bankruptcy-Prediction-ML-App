import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = pickle.load(open("model.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

st.set_page_config(page_title="Bankruptcy Prediction",layout="wide")

# ---------- CLEAN UI ----------
st.markdown("""
<style>
.stApp{
background: linear-gradient(135deg,#eef2f3,#dfe9f3);
}

.title{
text-align:center;
font-size:42px;
font-weight:bold;
color:#333;
}

.subtitle{
text-align:center;
color:#666;
}

.card{
background:white;
padding:25px;
border-radius:12px;
box-shadow:0 6px 20px rgba(0,0,0,0.15);
}

.safe{
background:#e8f5e9;
padding:15px;
border-radius:10px;
color:#2e7d32;
font-weight:bold;
text-align:center;
}

.risk{
background:#ffebee;
padding:15px;
border-radius:10px;
color:#c62828;
font-weight:bold;
text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown('<p class="title">Bankruptcy Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning model predicting company financial risk</p>', unsafe_allow_html=True)

st.write("")

# ---------- SIDEBAR INPUT ----------
st.sidebar.header("Company Risk Factors")

industrial_risk = st.sidebar.selectbox("Industrial Risk",[0,1])
management_risk = st.sidebar.selectbox("Management Risk",[0,1])
financial_flexibility = st.sidebar.selectbox("Financial Flexibility",[0,1])
credibility = st.sidebar.selectbox("Credibility",[0,1])
competitiveness = st.sidebar.selectbox("Competitiveness",[0,1])
operating_risk = st.sidebar.selectbox("Operating Risk",[0,1])

predict_button = st.sidebar.button("Predict Bankruptcy Risk")

# ---------- MODEL ACCURACY ----------
st.sidebar.markdown("### Model Accuracy")
st.sidebar.write("Logistic Regression : 100%")
st.sidebar.write("Random Forest : 100%")
st.sidebar.write("SVM : 100%")

# ---------- PREDICTION ----------
if predict_button:

    input_data = pd.DataFrame([[industrial_risk,
                                management_risk,
                                financial_flexibility,
                                credibility,
                                competitiveness,
                                operating_risk]],
                                columns=columns)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    col1,col2,col3 = st.columns(3)

    # ---------- METRICS ----------
    col1.metric("Bankruptcy Probability", str(round(probability*100,2))+"%")

    col2.metric("Risk Level",
                "HIGH RISK" if probability>0.6 else "LOW RISK")

    col3.metric("Model Confidence",
                str(round((1-abs(0.5-probability))*100,2))+"%")

    st.write("")

    # ---------- RESULT ----------
    if prediction[0] == 1:
        st.markdown('<div class="risk">⚠ Company is likely to go BANKRUPT</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="safe">✅ Company is FINANCIALLY SAFE</div>', unsafe_allow_html=True)

    st.write("")

    # ---------- GAUGE CHART ----------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text':"Bankruptcy Risk Meter"},
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':"red"},
            'steps':[
                {'range':[0,40],'color':"green"},
                {'range':[40,70],'color':"yellow"},
                {'range':[70,100],'color':"red"}
            ]
        }
    ))

    st.plotly_chart(gauge,use_container_width=True)

    # ---------- FEATURE IMPORTANCE ----------
    st.write("### Feature Importance")

    try:

        importance = model.coef_[0]

        fig,ax = plt.subplots()

        ax.barh(columns,importance)
        ax.set_xlabel("Impact")
        ax.set_title("Feature Impact on Bankruptcy")

        st.pyplot(fig)

    except:
        st.info("Feature importance not available for this model")

    # ---------- AI EXPLANATION ----------
    st.write("### AI Explanation")

    explanation = []

    values = [industrial_risk,
              management_risk,
              financial_flexibility,
              credibility,
              competitiveness,
              operating_risk]

    for name,value in zip(columns,values):

        if value==1:
            explanation.append(name.replace("_"," ").title())

    if len(explanation)>0:

        st.write("Risk factors detected:")

        for item in explanation:
            st.write("•",item)

    else:

        st.write("No major risk factors detected")

# ---------- FOOTER ----------
st.write("---")
st.caption("Developed using Machine Learning and Streamlit")
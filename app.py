import streamlit as st
import numpy as np
import joblib

# Page Config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# 🎨 Custom CSS (Modern UI)
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
}

/* Main Container */
.main {
    background: rgba(255, 255, 255, 0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

/* Title */
h1 {
    text-align: center;
    color: white;
    font-weight: bold;
}

/* Labels */
label {
    color: #e0e0e0 !important;
    font-weight: 500;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

/* Button Hover */
.stButton>button:hover {
    background: linear-gradient(90deg, #feb47b, #ff7e5f);
    transform: scale(1.05);
    transition: 0.3s;
}

/* Result Boxes */
.stSuccess {
    background-color: rgba(0, 255, 0, 0.2);
    color: white;
    border-radius: 10px;
}
.stError {
    background-color: rgba(255, 0, 0, 0.2);
    color: white;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# Load Model
model = joblib.load("heart_model.pkl")

# Header
st.markdown("<h1>🫀 Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.write("### 🤖 health risk detection")

st.markdown("---")

# Input Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", 20, 100)
    sex = st.selectbox("🚻 Sex (1=Male, 0=Female)", [1,0])
    cp = st.number_input("💢 Chest Pain Type (0-3)", 0, 3)
    trestbps = st.number_input("💓 Blood Pressure", 80, 200)
    chol = st.number_input("🧪 Cholesterol", 100, 400)
    fbs = st.selectbox("🍬 Fasting Blood Sugar (1=True, 0=False)", [1,0])
    restecg = st.selectbox("📊 Rest ECG (0-2)", [0,1,2])

with col2:
    thalach = st.number_input("🏃 Max Heart Rate", 60, 200)
    exang = st.selectbox("⚡ Exercise Angina (1=Yes, 0=No)", [1,0])
    oldpeak = st.number_input("📉 Oldpeak", 0.0, 6.0)
    slope = st.number_input("📈 Slope (0-2)", 0, 2)
    ca = st.number_input("🫀 Number of Vessels (0-3)", 0, 3)
    thal = st.number_input("🧬 Thal (1-3)", 1, 3)

st.markdown("---")

# Prediction
if st.button("🔍 Predict Heart Disease Risk"):

    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

    result = model.predict(data)
    prob = model.predict_proba(data)

    risk = prob[0][1] * 100  # probability of disease

    st.markdown("### 🧾 Prediction Result")

    # Prediction result
    if result[0] == 1:
        st.error("❌ Heart Disease Detected")
    else:
        st.success("✅ Heart Disease Not Detected")

    # Probability
    st.write(f"### 📊 Probability: {risk:.2f}%")

    # Risk Level
    if risk < 30:
        st.success("🟢 Low Risk")
    elif risk < 70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    # Progress bar
    st.progress(int(risk))

    # Suggestions
    st.markdown("### 💡 Health Suggestion")

    if risk > 70:
        st.error("⚠️ High risk! Please consult a doctor immediately.")
    elif risk > 40:
        st.warning("⚠️ Moderate risk. Improve lifestyle and diet.")
    else:
        st.success("✅ You are healthy. Maintain your lifestyle.")

# Footer
st.markdown("---")
st.markdown("### 👩‍💻 Developed by Anushka Mishra & Aaryan Kumar Ray")
st.caption("Machine Learning Project | BCA")
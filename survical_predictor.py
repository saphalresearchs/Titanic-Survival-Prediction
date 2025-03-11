import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load(open("model.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction")
st.write("Enter the passenger details below to predict survival.")

# Input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, step=1)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, step=1)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, step=1)
fare = st.number_input("Fare", min_value=0.0, step=0.1)
embarked = st.selectbox("Port of Embarkation", ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"])

# Encode categorical inputs
sex_encoded = 1 if sex == "Male" else 0
embarked_encoded = {"C (Cherbourg)": 1, "Q (Queenstown)": 2, "S (Southampton)": 0}[embarked]

# Make prediction
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction:")
    if prediction == 1:
        st.success("🎉 Survived! The passenger is likely to survive.")
    else:
        st.error("💔 Did not survive. The passenger is unlikely to survive.")

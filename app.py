import streamlit as st
import pandas as pd
import joblib

# Load your trained polynomial regression model
model = joblib.load('best_Polynomial_model.pkl')

st.title("Samsung Mobile Price Prediction App")

st.write("Enter Samsung mobile specifications below to predict its price (in PKR):")

# ===== Input fields for all features used in your dataset =====
build = st.number_input("Build (Version)", min_value=10, max_value=20, step=1)
sim = st.number_input("SIM (Count)", min_value=1, max_value=4, step=1)
processor = st.number_input("Processor (Version)", min_value=1, max_value=10, step=1)

display = st.selectbox("Display Type", ["Dynamic Amoled", "Ltpo Amoled", "Foldable"])
camera = st.number_input("Camera (MP)", min_value=1, max_value=200, step=1)
battery = st.number_input("Battery (mAh)", min_value=1000, max_value=7000, step=500)
storage = st.number_input("Storage (GB)", min_value=64, max_value=1024, step=64)
ram = st.number_input("RAM (GB)", min_value=4, max_value=24, step=1)
wifi_version = st.number_input("WiFi Version", min_value=5, max_value=7, step=1)

dual_band = st.selectbox("Dual Band", ["Yes", "No"])
tri_band = st.selectbox("Tri Band", ["Yes", "No"])
hotspot = st.selectbox("Hotspot", ["Yes", "No"])
wifi_direct = st.selectbox("WiFi Direct", ["Yes", "No"])
accelerometer = st.selectbox("Accelerometer", ["Yes", "No"])
compass = st.selectbox("Compass", ["Yes", "No"])
fingerprint = st.selectbox("Fingerprint Sensor", ["Yes", "No"])
barometer = st.selectbox("Barometer", ["Yes", "No"])
heartrate = st.selectbox("Heart Rate Sensor", ["Yes", "No"])

# ===== Convert categorical Yes/No to numeric =====
def yes_no(value):
    return 1 if value == "Yes" else 0

dual_band = yes_no(dual_band)
tri_band = yes_no(tri_band)
hotspot = yes_no(hotspot)
wifi_direct = yes_no(wifi_direct)
accelerometer = yes_no(accelerometer)
compass = yes_no(compass)
fingerprint = yes_no(fingerprint)
barometer = yes_no(barometer)
heartrate = yes_no(heartrate)

# Encode display type numerically (example mapping)
display_map = {"Dynamic Amoled": 0, "Ltpo Amoled": 1, "Foldable": 2}
display = display_map[display]

# ===== Predict Button =====
if st.button("Predict Price"):
    input_data = pd.DataFrame([[
        build, sim, processor, display, camera, battery, storage, ram,
        wifi_version, dual_band, tri_band, hotspot, wifi_direct,
        accelerometer, compass, fingerprint, barometer, heartrate
    ]], columns=[
        'Build', 'SIM', 'Processor', 'Display', 'Camera', 'Battery', 'Storage', 'RAM',
        'WiFi_Version', 'Dual_Band', 'Tri_Band', 'Hotspot', 'WiFi_Direct',
        'Accelerometer', 'Compass', 'Fingerprint', 'Barometer', 'HeartRate'
    ])
    
    price = model.predict(input_data)[0]
    st.success(f"Predicted Price: {price:.2f} PKR")

st.caption("Built using Polynomial Regression and Streamlit.")

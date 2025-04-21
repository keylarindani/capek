import streamlit as st
import pandas as pd
import pickle

# Load model
with open('best_xgboost_model (2).pkl', 'rb') as f:
    model = pickle.load(f)

# Load label encoders
with open('label_encoders (5).pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# App Title
st.title("Prediksi Pembatalan Booking Hotel 🏨")
st.write("Aplikasi ini memprediksi apakah sebuah booking hotel akan dibatalkan atau tidak menggunakan model XGBoost.")

# Form Input dari User
st.header("Masukkan Data Booking:")

type_of_meal_plan = st.selectbox("Tipe Meal Plan", label_encoders['type_of_meal_plan'].classes_)
room_type_reserved = st.selectbox("Tipe Kamar", label_encoders['room_type_reserved'].classes_)
market_segment_type = st.selectbox("Tipe Segmentasi Pasar", label_encoders['market_segment_type'].classes_)
required_car_parking_space = st.selectbox("Apakah memerlukan tempat parkir?", [0, 1])
avg_price_per_room = st.number_input("Harga Rata-Rata per Kamar", min_value=0.0, value=100.0, step=1.0)

# Tombol Prediksi
if st.button("Prediksi Booking"):
    # Encode input user
    encoded_input = {
        'type_of_meal_plan': [label_encoders['type_of_meal_plan'].transform([type_of_meal_plan])[0]],
        'room_type_reserved': [label_encoders['room_type_reserved'].transform([room_type_reserved])[0]],
        'market_segment_type': [label_encoders['market_segment_type'].transform([market_segment_type])[0]],
        'required_car_parking_space': [required_car_parking_space],
        'avg_price_per_room': [avg_price_per_room]
    }

    df_input = pd.DataFrame(encoded_input)

    # Prediksi
    prediction = model.predict(df_input)[0]
    result = "Canceled ❌" if prediction == 1 else "Not Canceled ✅"
    st.success(f"Prediksi: Booking akan **{result}**")

# ===========================================
# TEST CASE 1
st.header("🔍 Test Case 1")
if st.button("Jalankan Test Case 1"):
    test_case_1 = pd.DataFrame({
        'type_of_meal_plan': [label_encoders['type_of_meal_plan'].transform(['Meal Plan 1'])[0]],
        'room_type_reserved': [label_encoders['room_type_reserved'].transform(['Room_Type 1'])[0]],
        'market_segment_type': [label_encoders['market_segment_type'].transform(['Online'])[0]],
        'required_car_parking_space': [0],
        'avg_price_per_room': [120.0]
    })
    prediction1 = model.predict(test_case_1)[0]
    result1 = "Canceled ❌" if prediction1 == 1 else "Not Canceled ✅"
    st.info(f"Test Case 1: Booking diprediksi **{result1}**")

# TEST CASE 2
st.header("🔍 Test Case 2")
if st.button("Jalankan Test Case 2"):
    test_case_2 = pd.DataFrame({
        'type_of_meal_plan': [label_encoders['type_of_meal_plan'].transform(['Meal Plan 2'])[0]],
        'room_type_reserved': [label_encoders['room_type_reserved'].transform(['Room_Type 3'])[0]],
        'market_segment_type': [label_encoders['market_segment_type'].transform(['Offline'])[0]],
        'required_car_parking_space': [1],
        'avg_price_per_room': [80.0]
    })
    prediction2 = model.predict(test_case_2)[0]
    result2 = "Canceled ❌" if prediction2 == 1 else "Not Canceled ✅"
    st.info(f"Test Case 2: Booking diprediksi **{result2}**")

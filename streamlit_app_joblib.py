
import streamlit as st
import pandas as pd
import joblib

# Load model dan fitur
model = joblib.load("best_rf_model.joblib")
model_features = joblib.load("model_features.joblib")

# Judul aplikasi
st.title("Prediksi Harga Tiket Pesawat")

# Input pengguna
st.header("Masukkan Fitur Penerbangan")
airline = st.selectbox("Maskapai", ['Air_India', 'IndiGo', 'SpiceJet', 'Vistara', 'GO_FIRST'])
source_city = st.selectbox("Kota Asal", ['Delhi', 'Mumbai', 'Kolkata', 'Hyderabad', 'Bangalore', 'Chennai'])
destination_city = st.selectbox("Kota Tujuan", ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'Bangalore', 'Mumbai'])
departure_time = st.selectbox("Waktu Keberangkatan", ['Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night', 'Early_Morning'])
arrival_time = st.selectbox("Waktu Kedatangan", ['Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night', 'Early_Morning'])
stops = st.selectbox("Jumlah Transit", [0, 1, 2])
duration = st.slider("Durasi Penerbangan (jam)", 0, 50, 5)
days_left = st.slider("Hari Sebelum Keberangkatan", 1, 60, 30)
kelas = st.selectbox("Kelas Penerbangan", ['Economy', 'Business'])

# Proses input ke dalam DataFrame
input_dict = {
    'airline': airline,
    'source_city': source_city,
    'departure_time': departure_time,
    'stops': stops,
    'arrival_time': arrival_time,
    'destination_city': destination_city,
    'class': kelas,
    'duration': duration,
    'days_left': days_left
}

input_df = pd.DataFrame([input_dict])

# One-hot encoding agar cocok dengan model
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# Prediksi
if st.button("Prediksi Harga"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"Harga prediksi tiket: ₹ {int(prediction):,}")

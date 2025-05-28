
import streamlit as st
import pandas as pd
import pickle

# Load model dan fitur
with open("best_rf_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.title("🎫 Prediksi Harga Tiket Pesawat")

# Input pengguna
airline = st.selectbox("Maskapai", ["Air_India", "Indigo", "SpiceJet", "Vistara"])
source_city = st.selectbox("Kota Asal", ["Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai"])
departure_time = st.selectbox("Waktu Keberangkatan", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
stops = st.selectbox("Jumlah Transit", ["zero", "one", "two_or_more"])
arrival_time = st.selectbox("Waktu Tiba", ["Morning", "Afternoon", "Evening", "Night", "Late_Night", "Early_Morning"])
destination_city = st.selectbox("Kota Tujuan", ["Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai"])
duration = st.slider("Durasi Penerbangan (menit)", 30, 600, 120)
days_left = st.slider("Hari Sebelum Keberangkatan", 1, 60, 30)
kelas = st.radio("Kelas Penerbangan", ["Economy", "Business"])

# Buat input DataFrame satu baris
def make_input():
    base = {
        'duration': duration,
        'days_left': days_left,
        'is_business': 1 if kelas == "Business" else 0
    }
    for col in feature_names:
        if col not in base:
            base[col] = 0

    # Tambahkan one-hot
    base[f'airline_{airline}'] = 1
    base[f'source_city_{source_city}'] = 1
    base[f'departure_time_{departure_time}'] = 1
    base[f'stops_{stops}'] = 1
    base[f'arrival_time_{arrival_time}'] = 1
    base[f'destination_city_{destination_city}'] = 1

    # Susun ulang kolom
    df = pd.DataFrame([base])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

# Prediksi
if st.button("Prediksi"):
    input_df = make_input()
    pred = model.predict(input_df)[0]
    st.success(f"Harga prediksi tiket: ₹ {round(pred):,}")

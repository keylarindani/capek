import streamlit as st
import pandas as pd
import pickle

class HotelBookingApp:
    def __init__(self):
        self.model = self.load_pickle('best_xgboost_model (2).pkl')
        self.encoders = self.load_pickle('label_encoders (5).pkl')

    def load_pickle(self, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def encode_input(self, input_df):
        encoded_df = input_df.copy()
        for col in ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']:
            if col in self.encoders:
                encoded_df[col] = self.encoders[col].transform(encoded_df[col])
        return encoded_df

    def predict(self, input_df):
        encoded_df = self.encode_input(input_df)
        prediction = self.model.predict(encoded_df)[0]
        probability = self.model.predict_proba(encoded_df)[0][1]
        return prediction, probability

    def run(self):
        st.title("🏨 Prediksi Pembatalan Booking Hotel")
        st.write("Masukkan informasi booking untuk memprediksi apakah pemesanan **akan dibatalkan** atau **tidak**.")

        # Test Cases
        test_cases = {
            "Test Case 1": {
                'no_of_adults': 2,
                'no_of_children': 0,
                'no_of_weekend_nights': 1,
                'no_of_week_nights': 2,
                'type_of_meal_plan': 'Meal Plan 1',
                'required_car_parking_space': 0.0,
                'room_type_reserved': 'Room_Type 1',
                'lead_time': 45,
                'arrival_year': 2017,
                'arrival_month': 7,
                'arrival_date': 15,
                'market_segment_type': 'Online',
                'repeated_guest': 0,
                'no_of_previous_cancellations': 0,
                'no_of_previous_bookings_not_canceled': 0,
                'avg_price_per_room': 100.0,
                'no_of_special_requests': 1
            },
            "Test Case 2": {
                'no_of_adults': 1,
                'no_of_children': 2,
                'no_of_weekend_nights': 2,
                'no_of_week_nights': 5,
                'type_of_meal_plan': 'Meal Plan 2',
                'required_car_parking_space': 1.0,
                'room_type_reserved': 'Room_Type 3',
                'lead_time': 100,
                'arrival_year': 2017,
                'arrival_month': 12,
                'arrival_date': 25,
                'market_segment_type': 'Offline',
                'repeated_guest': 1,
                'no_of_previous_cancellations': 1,
                'no_of_previous_bookings_not_canceled': 3,
                'avg_price_per_room': 150.0,
                'no_of_special_requests': 2
            }
        }

        selected = st.selectbox("📁 Pilih Test Case", ["Manual Input"] + list(test_cases.keys()))

        if selected == "Manual Input":
            input_data = pd.DataFrame([{
                'no_of_adults': st.number_input('Jumlah Dewasa', 1, 10, 2),
                'no_of_children': st.number_input('Jumlah Anak-anak', 0, 10, 0),
                'no_of_weekend_nights': st.number_input('Malam Akhir Pekan', 0, 10, 1),
                'no_of_week_nights': st.number_input('Malam Hari Kerja', 0, 10, 2),
                'type_of_meal_plan': st.selectbox('Tipe Meal Plan', self.encoders['type_of_meal_plan'].classes_),
                'required_car_parking_space': float(st.selectbox('Butuh Tempat Parkir?', [0, 1])),
                'room_type_reserved': st.selectbox('Tipe Kamar', self.encoders['room_type_reserved'].classes_),
                'lead_time': st.slider('Lead Time (hari)', 0, 500, 45),
                'arrival_year': st.selectbox('Tahun Kedatangan', [2017, 2018]),
                'arrival_month': st.slider('Bulan Kedatangan', 1, 12, 7),
                'arrival_date': st.slider('Tanggal Kedatangan', 1, 31, 15),
                'market_segment_type': st.selectbox('Tipe Segmentasi Pasar', self.encoders['market_segment_type'].classes_),
                'repeated_guest': st.selectbox('Tamu Berulang?', [0, 1]),
                'no_of_previous_cancellations': st.slider('Jumlah Pembatalan Sebelumnya', 0, 10, 0),
                'no_of_previous_bookings_not_canceled': st.slider('Jumlah Pemesanan Sebelumnya yang Tidak Dibatalkan', 0, 10, 0),
                'avg_price_per_room': st.number_input('Harga Rata-rata Kamar', 0.0, 1000.0, 100.0),
                'no_of_special_requests': st.slider('Permintaan Khusus', 0, 5, 1)
            }])
        else:
            input_data = pd.DataFrame([test_cases[selected]])

        st.markdown("#### 📄 Data yang Akan Diprediksi")
        st.dataframe(input_data)

        if st.button("🔮 Prediksi"):
            prediction, probability = self.predict(input_data)
            status = "✅ Not Canceled" if prediction == 0 else "❌ Canceled"
            st.success(f"### Hasil Prediksi: {status}")
            st.info(f"### Probabilitas Pembatalan: {probability:.2%}")

if __name__ == "__main__":
    app = HotelBookingApp()
    app.run()

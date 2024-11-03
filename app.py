import streamlit as st
import joblib  # ใช้สำหรับโหลดโมเดลในรูปแบบ .joblib
import numpy as np

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('model.joblib')

# ส่วนติดต่อผู้ใช้ของ Streamlit
st.title("Prediction Web App")
st.write("ป้อนข้อมูลเพื่อทำนายผลลัพธ์ด้วยโมเดลที่โหลด")

# สร้างอินพุตสำหรับผู้ใช้ป้อนข้อมูล
feature_1 = st.text_area("ใส่ชื่อที่อยู่")
#feature_2 = st.number_input("Feature 2", min_value=0.0, max_value=100.0, value=50.0)

# เมื่อผู้ใช้กดปุ่ม "Predict"
if st.button("Predict"):
    # จัดรูปแบบข้อมูลอินพุตเป็น array
    input_data = np.array([[feature_1]])

    # ทำการพยากรณ์
    prediction = model.predict(input_data)

    # แสดงผลลัพธ์
    st.write("ผลลัพธ์ที่พยากรณ์ได้คือ:", prediction[0])

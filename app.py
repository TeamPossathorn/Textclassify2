import streamlit as st
import joblib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('model.joblib')

# ตั้งค่าหน้าเว็บแอป
st.title("Named Entity Recognition (NER) Prediction App")
st.write("กรุณาป้อนข้อมูลข้อความที่ต้องการให้โมเดลทำการตรวจจับ Named Entities")

# สร้างอินพุตให้ผู้ใช้ป้อนข้อความ
user_input = st.text_area("ป้อนข้อความที่นี่", "ใส่ข้อความที่คุณต้องการ")

# เมื่อผู้ใช้กดปุ่ม "Predict"
if st.button("Predict"):
    if user_input:
        # ทำการพยากรณ์โดยใช้โมเดล
        prediction = model.predict([user_input])
        
        # แยกคำจาก user_input
        words = user_input.split()

        # ตรวจสอบความยาวของคำและ labels
        if len(prediction[0]) == len(words):
            st.subheader("ผลลัพธ์การตรวจจับ Named Entities:")
            for i, label in enumerate(prediction[0]):
                st.write(f"คำที่ {i+1}: {words[i]} - {label}")
            
            # แสดงความถี่ของแต่ละประเภท
            st.subheader("ความถี่ของ Named Entities:")
            entity_counts = Counter(prediction[0])
            
            # แสดงข้อมูลความถี่ในรูปแบบตาราง
            st.write(entity_counts)
            
            # สร้างกราฟความถี่
            fig, ax = plt.subplots()
            ax.bar(entity_counts.keys(), entity_counts.values())
            ax.set_title("ความถี่ของ Named Entities")
            ax.set_xlabel("ประเภทของ Entity")
            ax.set_ylabel("ความถี่")
            st.pyplot(fig)
        else:
            st.error("เกิดข้อผิดพลาด: จำนวนคำในข้อความและผลการพยากรณ์ไม่ตรงกัน กรุณาป้อนข้อความใหม่หรือตรวจสอบโมเดล")
    else:
        st.warning("กรุณาป้อนข้อความก่อนทำการพยากรณ์")

import re
import streamlit as st
import joblib

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
        
        # ใช้ re.split เพื่อแยกคำด้วย " " หรือ "  "
        words = re.split(r' {1,2}', user_input)  # แยกด้วยช่องว่าง 1 หรือ 2 ช่อง
        
        # ตรวจสอบความยาวของคำใน user_input และ prediction[0]
        if len(prediction[0]) == len(words):
            st.subheader("ผลลัพธ์การตรวจจับ Named Entities:")
            for i, label in enumerate(prediction[0]):
                st.write(f"คำที่ {i+1}: {words[i]} - {label}")
        else:
            st.error("เกิดข้อผิดพลาด: จำนวนคำในข้อความและผลการพยากรณ์ไม่ตรงกัน กรุณาป้อนข้อความใหม่หรือตรวจสอบโมเดล")
    else:
        st.warning("กรุณาป้อนข้อความก่อนทำการพยากรณ์")

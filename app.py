import streamlit as st
import joblib
import random
from IPython.display import HTML
from collections import defaultdict

# โหลดโมเดล
model = joblib.load("model.joblib")

# กำหนดฟังก์ชันสร้างฟีเจอร์จาก tokens
stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

def tokens_to_features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace() if isinstance(word, str) else False,
        "word.is_stopword()": word in stopwords if isinstance(word, str) else False,
        "word.isdigit()": word.isdigit() if isinstance(word, str) else False,
        "word.islen5": word.isdigit() and len(word) == 5 if isinstance(word, str) else False
    }
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace() if isinstance(prevword, str) else False,
            "-1.word.is_stopword()": prevword in stopwords if isinstance(prevword, str) else False,
            "-1.word.isdigit()": prevword.isdigit() if isinstance(prevword, str) else False
        })
    else:
        features["BOS"] = True
    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace() if isinstance(nextword, str) else False,
            "+1.word.is_stopword()": nextword in stopwords if isinstance(nextword, str) else False,
            "+1.word.isdigit()": nextword.isdigit() if isinstance(nextword, str) else False
        })
    else:
        features["EOS"] = True
    return features

# ฟังก์ชันรันโมเดลและประมวลผลการทำนาย
def run_model(tokens):
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    predicted_tags = model.predict([features])[0]
    return predicted_tags

# เริ่มแอปพลิเคชัน Streamlit
st.title("Thai Address Tagging Application")
st.write("กรอกข้อความที่ต้องการประมวลผลแยกเป็น 5 ฟิลด์ แล้วดูผลลัพธ์")

# รับข้อความแยกกันเป็น 5 ฟิลด์
input_1 = st.text_input("Input Text 1", "")
input_2 = st.text_input("Input Text 2", "")
input_3 = st.text_input("Input Text 3", "")
input_4 = st.text_input("Input Text 4", "")
input_5 = st.text_input("Input Text 5", "")

# รวมข้อความทั้งหมดเป็นข้อความเดียว
full_text = f"{input_1} {input_2} {input_3} {input_4} {input_5}"

# ตรวจสอบเมื่อผู้ใช้กดปุ่มประมวลผล
if st.button("Run Model"):
    tokens = full_text.split()
    predicted_tags = run_model(tokens)

    # แสดงผลลัพธ์
    st.write("### Results")
    result_table = {"Token": tokens, "Predicted Tag": predicted_tags}
    for i, token in enumerate(tokens):
        st.write(f"{token}: {predicted_tags[i]}")

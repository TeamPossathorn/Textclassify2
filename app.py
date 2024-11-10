import streamlit as st
import joblib
import random
from IPython.display import HTML
from collections import defaultdict

# Load the model
model = joblib.load("model.joblib")

# Define stopwords and feature extraction function
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

# Function to run the model
def run_model(tokens, correct_tags):
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    predicted_tags = model.predict([features])[0]
    return predicted_tags

# Streamlit app
st.title("Thai Address Tagging Model")
st.write("ป้อนข้อมูลที่อยู่และให้โมเดลทำการแท็กส่วนต่างๆ")

# Inputs for the address
name_text = st.text_input("Name", "")
street_text = st.text_input("Street", "")
subdistrict_text = st.text_input("Subdistrict (Tambon)", "")
district_text = st.text_input("District (Amphoe)", "")
province_text = st.text_input("Province", "")
postal_code_text = st.text_input("Postal Code", "")

if st.button("Run Model"):
    full_address = f"{name_text} {street_text} {subdistrict_text} {district_text} {province_text} {postal_code_text}"
    tokens = full_address.split()

    correct_tags = []
    for token in name_text.split():
        correct_tags.append('O')
    for token in street_text.split():
        correct_tags.append('ADDR')
    for token in subdistrict_text.split():
        correct_tags.append('LOC')
    for token in district_text.split():
        correct_tags.append('LOC')
    for token in province_text.split():
        correct_tags.append('LOC')
    for token in postal_code_text.split():
        correct_tags.append('POST')

    if len(tokens) != len(correct_tags):
        st.error("Error: Number of tokens and tags do not match.")
    else:
        predicted_tags = run_model(tokens, correct_tags)
        result_table = {"Token": tokens, "Correct Tag": correct_tags, "Predicted Tag": predicted_tags}
        st.write("### Results")
        for i, token in enumerate(tokens):
            color = "red" if correct_tags[i] != predicted_tags[i] else "black"
            st.markdown(f"<span style='color:{color}'>{token} - {predicted_tags[i]}</span>", unsafe_allow_html=True)

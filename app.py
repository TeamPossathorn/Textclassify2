import streamlit as st
import joblib
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
import numpy as np
from sklearn_crfsuite import CRF
import plotly.graph_objects as go

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

# Function to run the model and display results
def run_model(tokens):
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    return model.predict([features])[0]

# Display Sankey Chart
def display_sankey_chart(data):
    input_labels = data["Input"]
    predict_labels = data["Predict Result"]
    correct_labels = data["Correct/Incorrect"]

    # Combine all unique labels
    all_labels = list(set(input_labels + predict_labels + correct_labels))

    # Sankey node indices
    source = [all_labels.index(x) for x in data["Input"]]
    target1 = [all_labels.index(x) for x in data["Predict Result"]]
    target2 = [all_labels.index(x) for x in data["Correct/Incorrect"]]

    # Sankey links
    source_links = source + target1
    target_links = target1 + target2
    value_links = [1] * len(source_links)  # Assign 1 for each link (example)

    # Create Sankey chart
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels
        ),
        link=dict(
            source=source_links,
            target=target_links,
            value=value_links
        )
    ))

    fig.update_layout(title_text="Sankey Diagram: Input vs Predict vs Correctness", font_size=10)
    st.plotly_chart(fig, use_container_width=True)

# Plot cumulative confusion matrix
def plot_cumulative_confusion_matrix():
    labels = sorted(set(st.session_state['all_true_tags']) | set(st.session_state['all_predicted_tags']))
    cm = confusion_matrix(st.session_state['all_true_tags'], st.session_state['all_predicted_tags'], labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=240)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Reds", ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot(fig)

# Sidebar Input Section
with st.sidebar:
    st.markdown("<p style='font-size:12px;'>Name</p>", unsafe_allow_html=True)
    name_text = st.text_input("", key="name", placeholder="Enter Name", help="Your name")

    st.markdown("<p style='font-size:12px;'>Street Address</p>", unsafe_allow_html=True)
    street_text = st.text_input("", key="street_address", placeholder="Enter Street Address", help="Your street address")

    st.markdown("<p style='font-size:12px;'>Province</p>", unsafe_allow_html=True)
    province_text = st.text_input("", key="province", placeholder="Enter Province", help="Your province")

    st.markdown("<p style='font-size:12px;'>Postal Code</p>", unsafe_allow_html=True)
    postal_code_text = st.text_input("", key="postal_code", placeholder="Enter Postal Code", help="Your postal code")

    if st.button("Run Model"):
        full_address = f"{name_text} {street_text} {province_text} {postal_code_text}"
        tokens = full_address.split()
        st.write(f"Tokens: {tokens}")  # Debugging information
        # Generate example data for Sankey chart
        data = {
            "Input": tokens,
            "Predict Result": ["LOC", "ADDR", "POST", "ADDR"],
            "Correct/Incorrect": ["Correct", "Incorrect", "Correct", "Incorrect"]
        }
        display_sankey_chart(data)

# Column 1: Entity Distribution
st.markdown("<h3 style='font-size:22px; color:black;'>Entity Distribution</h3>", unsafe_allow_html=True)
plot_cumulative_confusion_matrix()

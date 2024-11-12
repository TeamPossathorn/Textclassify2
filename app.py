import streamlit as st
import joblib
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pythainlp.tag.named_entity import ThaiNameTagger  # PyThaiNLP's NER module
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

# Load GitHub model
github_model_path = "path/to/github/model.joblib"  # Replace with actual model path

# Load GitHub model (replace with actual loading method)
try:
    github_model = joblib.load(github_model_path)
except Exception as e:
    st.error(f"Could not load GitHub model: {e}")

# Load PyThaiNLP NER model
thai_ner = ThaiNameTagger()

# Sample stopwords and feature extraction for GitHub model processing
stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

def tokens_to_features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5,
    }
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True
    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True
    return features

# Function to run GitHub model prediction
def run_github_model(tokens, model):
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    return model.predict([features])[0]

# Function to run PyThaiNLP NER model
def run_thai_ner(text):
    return thai_ner.get_ner(text)

# Visualization functions
def display_entities(tokens, predicted_tags):
    result_html = ""
    for token, tag in zip(tokens, predicted_tags):
        color = "blue" if tag == "O" else "green"
        result_html += f"<span style='color:{color}'>{token} ({tag})</span> "
    st.markdown(result_html, unsafe_allow_html=True)

def plot_confusion_matrix(correct_tags, predicted_tags):
    labels = sorted(set(correct_tags) | set(predicted_tags))
    cm = confusion_matrix(correct_tags, predicted_tags, labels=labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

def plot_entity_distribution(predicted_tags, model_name):
    entity_counts = Counter(predicted_tags)
    labels, values = zip(*entity_counts.items())
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.title(f"Entity Distribution in {model_name} Predictions")
    st.pyplot(fig)

def display_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    st.write("### Classification Report")
    st.dataframe(report)

# Streamlit UI
st.title("NER Visualization with Two Models")
st.write("Compare GitHub Model and PyThaiNLP NER Model")

# User input text
input_text = st.text_area("Enter text for NER")

if st.button("Run Models"):
    tokens = input_text.split()

    # Run GitHub model and display results
    st.write("### GitHub Model Predictions")
    github_predictions = run_github_model(tokens, github_model)
    display_entities(tokens, github_predictions)
    plot_entity_distribution(github_predictions, "GitHub Model")

    # Run PyThaiNLP NER model and display results
    st.write("### PyThaiNLP Model Predictions")
    thai_ner_results = run_thai_ner(input_text)
    thai_predictions = [tag for _, tag in thai_ner_results]  # Extract entity tags only
    display_entities([word for word, _ in thai_ner_results], thai_predictions)
    plot_entity_distribution(thai_predictions, "PyThaiNLP Model")

    # Display Confusion Matrix for GitHub Model vs PyThaiNLP Model
    st.write("### Confusion Matrix (GitHub Model vs PyThaiNLP Model)")
    plot_confusion_matrix(github_predictions, thai_predictions)

    # Display Classification Report
    st.write("### Classification Report for GitHub Model")
    display_classification_report(github_predictions, tokens)

    st.write("### Classification Report for PyThaiNLP Model")
    display_classification_report(thai_predictions, tokens)

import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from scipy.sparse import hstack, csr_matrix
import re
import pickle
from urllib.parse import urlparse
import whois
from datetime import datetime

# Load Model and Vectorizer
with open('phishing_url_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('phishing_url_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing Function
def preprocess_url(url):
    return re.sub(r'[^A-Za-z0-9]', ' ', url)

# Feature Extraction Function
def extract_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    suspicious_keywords = ['login', 'verify', 'bank', 'update', 'security', 'account']
    suspicious_tlds = ['.tk', '.ga', '.ml', '.cf', '.gq']

    try:
        domain_info = whois(domain)
        domain_age = (datetime.now() - domain_info.creation_date[0]).days if domain_info.creation_date else 0
    except:
        domain_age = 0

    return {
        'url_length': len(url),
        'domain_length': len(domain),
        'path_length': len(path),
        'num_subdomains': domain.count('.'),
        'has_ip': int(bool(re.search(r'\b\d{1,3}(\.\d{1,3}){3}\b', domain))),
        'has_suspicious_tld': int(any(tld in domain for tld in suspicious_tlds)),
        'domain_age': domain_age,
        'is_blacklisted': int('blacklist' in url.lower()),
        'has_https': int(parsed_url.scheme == 'https'),
        'num_special_chars': sum(url.count(c) for c in '@?-=_'),
        'num_digits': sum(c.isdigit() for c in url),
        'num_params': url.count('&'),
        'has_suspicious_keywords': int(any(keyword in url.lower() for keyword in suspicious_keywords)),
    }

# Prediction Function
def predict_url(url):
    processed_url = preprocess_url(url)
    features = extract_features(url)
    features_sparse = csr_matrix([list(features.values())])
    vectorized_url = vectorizer.transform([processed_url])
    combined_features = hstack([features_sparse, vectorized_url])
    prediction = rf_model.predict(combined_features)
    prob = rf_model.predict_proba(combined_features)[:, 1][0]
    return ("Phishing" if prediction[0] == 1 else "Legitimate", prob)

# Streamlit App
st.title("Phishing URL Detection System")

# URL Prediction Section
st.header("Enter a URL to Check")
user_url = st.text_input("URL:")

if st.button("Check URL"):
    if user_url:
        result, score = predict_url(user_url)
        st.success(f"Result: {result}")
        st.info(f"Severity Score: {score:.2%}")
    else:
        st.error("Please enter a valid URL.")

# Model Evaluation Section
st.header("Model Evaluation")

# Dummy data for visualization (replace with actual test data and predictions)
# Ensure 'y_test' and 'y_pred' are properly populated for meaningful results.
y_test = np.array([0, 1, 0, 1, 0])  # Replace with actual labels
y_pred = np.array([0, 1, 1, 1, 0])  # Replace with actual predictions

# Confusion Matrix
st.subheader("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', ax=ax)
ax.set_title("Confusion Matrix")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")
class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
st.dataframe(class_report_df)

import streamlit as st
from transformers import pipeline
import torch
import os
import boto3

BUCKET_NAME = "mlops-kgptalkie-faraz"

s3 = boto3.client('s3')

def download_dir(local_dir, s3_prefix):
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                local_file_path = os.path.join(local_dir, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3.download_file(BUCKET_NAME, s3_key, local_file_path)

st.title("Sentiment Analysis")

local_dir = 'tinybert-sentiment-analysis'
s3_prefix = "ml-models/tinybert-sentiment-analysis/"

download_model_button = st.button("Download Model")

if download_model_button:
    with st.spinner("Downloading... Please wait"):
        download_dir(local_dir, s3_prefix)
    st.success("Model downloaded!")

review = st.text_area("Enter your review", "Type...")

device = 0 if torch.cuda.is_available() else -1

classifier = None
if os.path.exists(local_dir):
    classifier = pipeline("text-classification", model=local_dir, device=device)

predict_button = st.button("Predict")

if predict_button and classifier:
    with st.spinner("Predicting..."):
        output = classifier(review)
        st.write(output)
elif predict_button:
    st.error("Model not loaded. Please download it first.")
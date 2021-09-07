from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import json
import os

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


# download label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode("utf-8").split("\n")
    csvreader = csv.reader(html, delimiter="\t")
labels = [row[1] for row in csvreader if len(row) > 1]


def model_fn(model_dir):
    print("********* model_dir path**********", model_dir)
    print(os.listdir(model_dir))
    model_path = os.path.join(model_dir, "model_token")
    print("********* list directory **********", os.listdir(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return [model, tokenizer]


def input_fn(text, context):
    text = preprocess(text)
    print("**************** Data ***************", text)
    return text


def predict_fn(text, model):
    model_1, tokenizer = model
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model_1(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    data = {}
    data["result"] = []
    for i in range(scores.shape[0]):
        data["result"].append({"score": str(scores[ranking[i]]), "labels": str(labels[ranking[i]])})
    json_data = json.dumps(data)
    print("************** predict_fn end *******************")
    return json_data

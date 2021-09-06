from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import json
import os
os.system('pip install retrying')

from retrying import retry

# define output data variable
data = {}
data['predictions'] = []

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/mapping.txt"


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

def model_fn(model_dir):
    print('********* model_dir path**********', model_dir)
    print(os.listdir(model_dir))
    model_path = os.path.join(model_dir,'model_token')
    print('********* list directory **********', os.listdir(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return [model, tokenizer]


@retry(stop_max_attempt_number=5, wait_fixed=5000)
def input_fn(text, context):
    print('**************** Start input_fn ***************')
    print(type(text))
    try:
        processed_text = json.loads(text)
    except: 
        print("*********** Unable to load the json file, trying again *********")
    print(type(processed_text))
    print('**************** End input_fn ***************')
    return processed_text

def predict_fn(corpus, model):
    print('**************** Start predict_fn ***************')
    model_1, tokenizer = model
    for text in corpus['instances']:
        processed_text = preprocess(text['inputs'])
        print(processed_text)
        encoded_input = tokenizer(str(processed_text), return_tensors='pt')
        output = model_1(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        sentence = {}
        sentence['text'] = []
        for i in range(scores.shape[0]):
            sentence['text'].append({
                'score': str(scores[ranking[i]]), 
                'label': str(labels[ranking[i]])
            })
        data['predictions'].append(sentence)
    json_data = json.dumps(data)
    print("************** predict_fn end *******************")
    return json_data
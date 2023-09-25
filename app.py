#pip install fastapi uvicorn

# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import numpy as np

# Create the app object
app = FastAPI()


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)


model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch_device)

tokenizer = AutoTokenizer.from_pretrained(model_name)



# Index route, opens automatically on http://127.0.0.1:8000

@app.get('/')
def home():
    return {"message": "Hello World"}


@app.get('/predict')
def predict_title(title: str):
    encoded_input = tokenizer(title, return_tensors='pt')
    with torch.no_grad():
        prediction = model(**encoded_input)
    

    prediction_value = int(np.argmax(prediction['logits'].numpy()))

    return {'predicted class':  prediction_value }


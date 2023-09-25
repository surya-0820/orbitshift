#pip install fastapi uvicorn

# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import numpy as np

# 2. Create the app object
app = FastAPI()


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)


model_name = "D:\orbit_project\my_model"  
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.to(torch_device)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/predict')
def predict_title(title: str):
    encoded_input = tokenizer(title, return_tensors='pt')
    with torch.no_grad():
        prediction = model(**encoded_input)
    

    prediction_value = int(np.argmax(prediction['logits'].numpy()))

    return {'predicted class':  prediction_value }



# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn app:app --reload
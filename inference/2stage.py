import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Prepare input texts. This model is pretrained on Czech data 
test_texts = ['Utterance1;Utterance2;Utterance3']

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    'justtherightsize/robeczech-2stage-online-risks-cs', num_labels=5).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    'justtherightsize/robeczech-2stage-online-risks-cs',
    use_fast=False, truncation_side='left')
assert tokenizer.truncation_side == 'left'

# Define helper functions
def predict_one(text: str, tok, mod, threshold=0.5):
    encoding = tok(text, return_tensors="pt", truncation=True, padding=True,
                   max_length=256)
    encoding = {k: v.to(mod.device) for k, v in encoding.items()}
    outputs = mod(**encoding)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= threshold)] = 1
    return predictions, probs

def print_predictions(texts):
    preds = [predict_one(tt, tokenizer, model) for tt in texts]
    for c, p in preds:
        print(f'{c}: {p.tolist():.4f}')

# Run the prediction
print_predictions(test_texts)
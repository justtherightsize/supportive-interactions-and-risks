import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Prepare input texts. This model is fine-tuned for Czech.
test_texts = ['utterance1;utterance2;utterance3']

# Load the model and tokenizer
bin_model_name = 'justtherightsize/small-e-czech-binary-online-risks-cs'
stage2_model_name = 'justtherightsize/small-e-czech-2stage-online-risks-cs'
bin_model = AutoModelForSequenceClassification.from_pretrained(
    bin_model_name, num_labels=2).to("cuda")
stage2_model = AutoModelForSequenceClassification.from_pretrained(
    stage2_model_name, num_labels=5).to("cuda")
bin_tokenizer = AutoTokenizer.from_pretrained(
    bin_model_name, use_fast=False, truncation_side='left')
stage2_tokenizer = AutoTokenizer.from_pretrained(
    stage2_model_name, use_fast=False, truncation_side='left')
assert bin_tokenizer.truncation_side == 'left'
assert stage2_tokenizer.truncation_side == 'left'

# Define helper functions
def get_preds_bin(text, tokenizer, model):
    """
    Return the predicted classes
    """
    inputs = tokenizer(text, padding=True, truncation=True, max_length=256,
                       return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    preds = torch.argmax(probs, -1).detach().cpu()
    return preds

def get_preds_multi(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=256,
                       return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs.detach().cpu() >= 0.5)] = 1
    return predictions

bin_decoded = get_preds_bin(test_texts, bin_tokenizer, bin_model)
stage2_decoded = get_preds_multi(test_texts, stage2_tokenizer, stage2_model)

preds_combined = []
for b,m in zip(bin_decoded, stage2_decoded):
    if b == 0:
        preds_combined.append([1,0,0,0,0,0])
    else:
        preds_combined.append([0] + m.astype(int).tolist())

print(preds_combined)
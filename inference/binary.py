import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Prepare input texts. This model is fine-tuned for Czech
test_texts = ['Utterance1;Utterance2;Utterance3']

# Load the model and tokenizer
model_name = 'justtherightsize/robeczech-binary-online-risks-cs'

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=False, truncation_side='left')
assert tokenizer.truncation_side == 'left'

# Define helper functions
def get_probs(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=256,
                       return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    return outputs[0].softmax(1)

def preds2class(probs, threshold=0.5):
    pclasses = np.zeros(probs.shape)
    pclasses[np.where(probs >= threshold)] = 1
    return pclasses.argmax(-1)

def print_predictions(texts):
    probabilities = [get_probs(
        texts[i], tokenizer, model).cpu().detach().numpy()[0]
                     for i in range(len(texts))]
    predicted_classes = preds2class(np.array(probabilities))
    for c, p in zip(predicted_classes, probabilities):
        print(f'{c}: {p}')

# Run the prediction
print_predictions(test_texts)
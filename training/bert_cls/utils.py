import json
import os
import sys
from pathlib import Path
import random
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import logging as log
import torch
import transformers
import wandb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, \
    multilabel_confusion_matrix, classification_report
from sklearn.utils import shuffle
from torch import nn
from transformers import AutoTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments, \
    EarlyStoppingCallback, is_torch_available, is_tf_available, RobertaForSequenceClassification, \
    BertForSequenceClassification, XLMRobertaForSequenceClassification, EvalPrediction, \
    XLMRobertaXLForSequenceClassification


def set_gpus(devices: str):
    """
    @param devices: String of shape: '0,1,...'
    """
    no_gpu = len(devices.split(','))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    print('No. of devices: ' + str(no_gpu) + ' : ' + devices)
    return no_gpu


def set_seed(seed: int):
    """
    Set reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


def rminsidedir(directory, prefix: str):
    directory = Path(directory)
    with os.scandir(directory) as d:
        for item in d:
            if item.name.startswith(prefix):
                rmdir(item)


def load_config(file: Path) -> dict:
    with open(file) as json_data:
        root = json.load(json_data)
    return root


def mk_out_dir(run_id: str, embed_in_dir: str) -> Path:
    if embed_in_dir:
        pth = Path('results', run_id, embed_in_dir)
    else:
        pth = Path('results', run_id)
    os.makedirs(pth, exist_ok=True)
    return pth



def predict_one(text: str, tok, mod):
    encoding = tok(text, return_tensors="pt")
    encoding = {k: v.to(mod.device) for k, v in encoding.items()}
    outputs = mod(**encoding)
    logits = outputs.logits
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [idx for idx, label in enumerate(predictions) if label == 1.0]
    print(f'{predicted_labels}, probs: {probs}')


def predict_one_ml(text: str, tok, mod, threshold=0.5):
    encoding = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=ARGS['max_length'])
    encoding = {k: v.to(mod.device) for k, v in encoding.items()}
    outputs = mod(**encoding)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= threshold)] = 1
    return predictions


def predict_one_m2b(text: str, tok, mod):
    encoding = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=ARGS['max_length'])
    encoding = {k: v.to(mod.device) for k, v in encoding.items()}
    outputs = mod(**encoding)
    logits = outputs.logits
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [idx for idx, label in enumerate(predictions) if label == 1.0]
    return 1 if sum(predicted_labels) > 0 else 0


def plot_loss(output_dir: str, loss, f1):
    plt.plot(loss)
    plt.plot(f1)
    plt.savefig(Path(output_dir, 'loss.png'))  # out dir must be specified before
    plt.show()  # needs to be after save, bc it deletes the plot


def load_model_bin(model_name: str, target_names: str):
    log.info('Load model and move to GPU...')
    if model_name == 'Seznam/small-e-czech':
        model = ElectraForSequenceClassification.from_pretrained(
            model_name, num_labels=len(target_names)).to("cuda")
    elif model_name == 'ufal/robeczech-base':
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=len(target_names)).to("cuda")
    elif model_name == 'xlm-roberta-base' or model_name == 'xlm-roberta-large':
        model = XLMRobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=len(target_names)).to("cuda")
    elif model_name == 'facebook/xlm-roberta-xl':
        model = XLMRobertaXLForSequenceClassification.from_pretrained(
            model_name, num_labels=len(target_names)).to("cuda")
    elif model_name == 'roberta-base':
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=len(target_names)).to("cuda")
    else:
        raise Exception('Unknown model name.')
    return model



def get_inverse_class_ratio_bin(labels):
    # configure class weight by pos/all ratio
    sm = sum(labels)
    pos = sm / len(labels)
    neg = 1 - pos
    # reverse pos/neg to offset weights
    class_ratio = [pos, neg]
    return class_ratio


def oversample_coarse_half(train_texts, train_labels):
    """
    # oversample the minority class
    """
    pos_ratio, neg_ratio = get_inverse_class_ratio_bin(train_labels)
    log.info('Class weights before ovesample (pos:neg) = {:.2f}:{:.2f}'.format(pos_ratio, neg_ratio))
    minority_ratio = min(pos_ratio, neg_ratio)
    minority_label = int(np.argmin([neg_ratio, pos_ratio]))
    # first try to multiplicate all of them
    if minority_ratio * 1.95 <= max(pos_ratio, neg_ratio):
        random.seed(42)
        os_texts = []
        os_labels = []
        multiplier = 1 / minority_ratio
        add_n = int(multiplier) - 1
        for i, v in enumerate(train_labels):
            if v != minority_label:
                continue
            for j in range(add_n):
                os_labels.append(v)
                os_texts.append(train_texts[i])
        train_texts.extend(os_texts)
        train_labels.extend(os_labels)
    # add more randomly selected minorities to result in 50:50 ratio
    pos_ratio, neg_ratio = get_inverse_class_ratio_bin(train_labels)
    minority_ratio = min(pos_ratio, neg_ratio)
    majority_ratio = max(pos_ratio, neg_ratio)
    minority_label = int(np.argmin([neg_ratio, pos_ratio]))
    to_add = int(len(train_labels) * (majority_ratio - minority_ratio))
    os_texts = []
    os_labels = []
    added = 0
    while added < to_add:
        idx = random.randint(0, len(train_labels) - 1)
        if train_labels[idx] == minority_label:
            os_labels.append(train_labels[idx])
            os_texts.append(train_texts[idx])
            added += 1
    train_texts.extend(os_texts)
    train_labels.extend(os_labels)
    train_texts, train_labels = shuffle(train_texts, train_labels)
    log.info(f'Oversampled class: {minority_label}')
    clr = get_inverse_class_ratio_bin(train_labels)
    log.info('Class weights after ovesample (pos:neg): {:.2f}:{:.2f}'.format(clr[0], clr[1]))
    return train_texts, train_labels, clr


def oversample_multiclass_equal(train_texts, train_labels):
    lbls = np.array(train_labels)
    counts = lbls.sum(axis=0)
    log.info(f'Class counts before oversample: {counts}')

    max_idx = counts.argmax(axis=0)
    to_add = (1 - (counts / counts[max_idx])) * counts[max_idx]

    os_texts = []
    os_labels = []
    while to_add.sum() > 0:
        idx = random.randint(0, len(train_labels) - 1)
        l = lbls[idx].argmax(axis=0)
        if l != max_idx and to_add[l] > 0:
            os_labels.append(train_labels[idx])
            os_texts.append(train_texts[idx])
            to_add[l] -= 1
    train_texts.extend(os_texts)
    train_labels.extend(os_labels)
    train_texts, train_labels = shuffle(train_texts, train_labels)
    lbls = np.array(train_labels)
    counts = lbls.sum(axis=0)
    log.info(f'Class counts before oversample: {counts}')
    return train_texts, train_labels, counts


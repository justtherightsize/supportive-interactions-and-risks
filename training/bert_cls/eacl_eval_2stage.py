import json
import sys
from pathlib import Path
import logging as log

import numpy as np
import torch
import transformers
import wandb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    multilabel_confusion_matrix, classification_report
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, EvalPrediction, \
    ElectraForSequenceClassification, RobertaForSequenceClassification

from bert_cls.utils import load_config, mk_out_dir, set_seed, set_gpus, rminsidedir, load_model_bin, \
    oversample_multiclass_equal


def transform_and_filter(coarse_texts, coarse_labels, multi_labels):
    """Filters out 0 labels and multi-label anntations and cuts off the 0 label"""
    tt = []
    tlc = []
    tlm = []
    for t, cl, ml in zip(coarse_texts, coarse_labels, multi_labels):
        if sum(ml) == 1:
            tt.append(t)
            tlc.append(cl)
            tlm.append(ml)
    return tt, tlc, tlm


def read_split_data(tag: str, dirr: Path, split_i: str):
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.test')) as json_data:
        root = json.load(json_data)
        coarse_texts, coarse_labels, multi_labels = transform_and_filter(
            [x[0] for x in root['coarse']],
            [x[1] for x in root['coarse']],
            [x[1] for x in root['multi']])
    return coarse_texts, coarse_labels, multi_labels


def get_preds_bin(text, tokenizer, model):
    """
    Return the predicted classes
    """
    inputs = tokenizer(text, padding=True, truncation=True, max_length=ARGS['max_length'],
                       return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    preds = torch.argmax(probs, -1).detach().cpu()
    return preds


def get_preds_multi(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=ARGS['max_length'],
                       return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    preds = torch.argmax(probs, -1).detach().cpu()
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs.detach().cpu() >= 0.5)] = 1
    return predictions


def run():
    log.info('Loading datasets...')
    texts, coarse_labels, multi_labels = read_split_data(ARGS['tag'], ARGS['dir'], ARGS['split'])
    texts = texts[:500]
    coarse_labels = coarse_labels[:500]
    multi_labels = multi_labels[:500]

    log.info('Predict coarse...')
    test_batch_size = 64
    model_bin_pth = Path('results', ARGS['run_id'], ARGS['split'], 'split-{}_{}_id-{}'.format(ARGS['split'], ARGS['tag'], ARGS['run_id'])) #TODO
    tokenizer = AutoTokenizer.from_pretrained(model_bin_pth, use_fast=False, truncation_side='left')
    assert tokenizer.truncation_side == 'left'
    model_bin = ElectraForSequenceClassification.from_pretrained(model_bin_pth, num_labels=len(ARGS['target_names'])).to(
        "cuda")
    preds = []
    for batch_offset in tqdm(range(0, len(coarse_labels), test_batch_size)):
        input_text_batch = [text for text in texts[batch_offset:batch_offset + test_batch_size]]
        decoded = get_preds_bin(input_text_batch, tokenizer, model_bin)
        preds.extend(decoded)


    log.info('Predict multiclass...')
    model_mc_pth = Path('results', ARGS2['run_id'], ARGS['split'], 'split-{}_{}_id-{}'.format(ARGS['split'], ARGS2['tag'], ARGS2['run_id'])) #TODO
    tokenizer = AutoTokenizer.from_pretrained(model_mc_pth, use_fast=False, truncation_side='left')
    assert tokenizer.truncation_side == 'left'
    model_mc = ElectraForSequenceClassification.from_pretrained(model_mc_pth, num_labels=len(ARGS2['target_names'])).to(
        "cuda")
    preds_mc = []
    for batch_offset in tqdm(range(0, len(coarse_labels), test_batch_size)):
        input_text_batch = [text for text in texts[batch_offset:batch_offset + test_batch_size]]
        decoded = get_preds_multi(input_text_batch, tokenizer, model_mc)
        preds_mc.extend(decoded)

    # construct the multiclass labels
    preds_combined = []
    for b,m in zip(preds, preds_mc):
        if b == 0:
            preds_combined.append([1,0,0,0,0,0])
        else:
            preds_combined.append([0] + m.astype(int).tolist())

    cr = classification_report(multi_labels, preds_combined, output_dict=True)
    log.info(cr)
    # cfm = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(preds, axis=1)).tolist()
    results_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}_2stage.json'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    with open(results_pth, 'w', encoding='utf-8') as outfile:
        ARGS['output_dir'] = str(ARGS['output_dir'])  # need to convert Path to str before dump
        result = {
            # 'micro_f1': str(multi_f1(test_labels, preds, method="micro")),
            # 'macro_f1': str(multi_f1(test_labels, preds, method="macro")),
            # 'weighted_f1': str(multi_f1(test_labels, preds, method="weighted")),
            # 'test_cfm': str(cfm),
            'cr': str(cr)
            # 'args': ARGS
        }
        # result['args']['class_ratio'] = result['args']['class_ratio'].tolist()
        json.dump(result, outfile, ensure_ascii=False)
    # rminsidedir(ARGS['output_dir'], 'checkpoint')
    # log.info('Finished.')


def main():
    # set seed and GPUs
    set_seed(1)
    no_gpus = set_gpus(ARGS['gpu'])
    ARGS['visible_devices'] = no_gpus

    # run
    run()  # results are saved in split_# dir


if __name__ == '__main__':
    # load args
    ARGS = load_config(Path(sys.argv[1]))
    ARGS2 = load_config(Path(sys.argv[2]))
    ARGS['split'] = str(sys.argv[3])  # split_#
    ARGS['gpu'] = str(sys.argv[4])  # single string number
    ARGS['output_dir'] = mk_out_dir(ARGS['run_id'], ARGS['split'])  # results/<run_id>/<split_#>

    # create global loss logger
    LOSS = []
    F1 = []
    CFM = []

    # file logging
    log.basicConfig(
        level=log.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[log.FileHandler(filename=Path(ARGS['output_dir'],
                                                'split-{}_{}_id-{}.log'.format(ARGS['split'],
                                                                               ARGS['tag'],
                                                                               ARGS['run_id']))),
                  log.StreamHandler(sys.stdout)]
    )
    # wandb.init(project="irtisdb")
    wandb.init(mode="disabled")
    # run
    try:
        main()
    except Exception as e:
        log.exception(e)
        raise

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
    BertForSequenceClassification, XLMRobertaForSequenceClassification, EvalPrediction

from bert_cls.utils import set_seed, set_gpus


def oversample_multi_half(train_texts, train_labels):
    sumz = [0,0,0,0,0,0]
    for a in train_labels:
        for i, b in enumerate(a):
            sumz[i] += b
    countx = sumz[0] // 5

    new_texts = []
    new_labels = []
    rng1 = range((countx-sumz[1]) // sumz[1])
    rng2 = range((countx-sumz[2]) // sumz[2])
    rng3 = range((countx-sumz[3]) // sumz[3])
    rng4 = range((countx-sumz[4]) // sumz[4])
    rng5 = range((countx-sumz[5]) // sumz[5])

    for t,l in zip(train_texts, train_labels):
        if l[1] == 1:
            for x in rng1:
                new_texts.append(t)
                new_labels.append(l)
        if l[2] == 1:
            for x in rng2:
                new_texts.append(t)
                new_labels.append(l)
        if l[3] == 1:
            for x in rng3:
                new_texts.append(t)
                new_labels.append(l)
        if l[4] == 1:
            for x in rng4:
                new_texts.append(t)
                new_labels.append(l)
        if l[5] == 1:
            for x in rng5:
                new_texts.append(t)
                new_labels.append(l)
    assert len(new_texts) == len(new_labels)
    new_texts, new_labels = shuffle(train_texts + new_texts, train_labels + new_labels)
    assert len(new_texts) == len(new_labels)
    return new_texts, new_labels


def read_split_data(tag: str, dirr: Path, split_i: str, oversample_train="none"):
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.train')) as json_data:
        root = json.load(json_data)
        train_texts = [x[0] for x in root['multi']]
        train_labels = [x[1] for x in root['multi']]

        if oversample_train == "none":
            pass
        elif oversample_train == "half":
            train_texts, train_labels = oversample_multi_half(train_texts, train_labels)
        else:
            raise NotImplemented

    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.dev')) as json_data:
        root = json.load(json_data)
        dev_texts = [x[0] for x in root['multi']]
        dev_labels = [x[1] for x in root['multi']]

    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.test')) as json_data:
        root = json.load(json_data)
        test_texts = [x[0] for x in root['multi']]
        test_labels = [x[1] for x in root['multi']]

    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


def read_coarse_test_data(tag: str, dirr: Path, split_i: str):
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.test')) as json_data:
        root = json.load(json_data)
        test_texts = [x[0] for x in root['coarse']]
        test_labels = [x[1] for x in root['coarse']]
    return test_texts, test_labels


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    # roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_macro_average,
               'f1_micro': f1_micro_average,
               'f1_weighted': f1_weighted_average,
               # 'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


def get_probs(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=ARGS['max_length'],
                       return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    return outputs[0].softmax(1)


def load_config(file: Path) -> dict:
    with open(file) as json_data:
        root = json.load(json_data)
    return root


def mk_out_dir(embed_in_dir: str) -> Path:
    if embed_in_dir:
        pth = Path('results', ARGS['run_id'], embed_in_dir)
    else:
        pth = Path('results', str(ARGS['run_id']))
    os.makedirs(pth, exist_ok=True)
    return pth


def plot_loss():
    plt.plot(LOSS)
    plt.plot(F1)
    plt.savefig(Path(ARGS['output_dir'], 'loss.png'))  # out dir must be specified before
    plt.show()  # needs to be after save, bc it deletes the plot


class EncodingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)


class LogCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if 'loss' in state.log_history[-2]:
            LOSS.append(state.log_history[-2]['loss'])
        if 'eval_f1' in state.log_history[-1]:
            F1.append(state.log_history[-1]['eval_f1'])
        # if 'eval_cfm' in state.log_history[-1]:
        #     CFM.append(state.log_history[-1]['eval_cfm'])


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


def bin_f1(true_labels, pred_labels):
    return round(f1_score(y_true=true_labels, y_pred=pred_labels), 4)


def multi_f1(true_labels, pred_labels, method):
    return round(f1_score(y_true=true_labels, y_pred=pred_labels, average=method), 4)


def bootstrap_michal(true_labels, pred_labels, threshold=0.5):
    bootstrap_sample = int(len(true_labels)/10)
    bootstrap_repeats = 50
    evaluations = []
    ret = {}
    for fscore_method in ["weighted", "micro", "macro"]:
        for repeat in range(bootstrap_repeats):
            sample_idx = random.sample(list(range(len(pred_labels))), k=bootstrap_sample)
            pred_sample = [pred_labels[idx] for idx in sample_idx]
            true_sample = [true_labels[idx] for idx in sample_idx]
            evaluation = f1_score(true_sample, pred_sample, average=fscore_method)
            evaluations.append(evaluation)

        lower = np.quantile(evaluations, q=0.025)
        upper = np.quantile(evaluations, q=0.975)
        mean = np.mean(evaluations)
        log.info("Coarse-seq F-scores (%s) interval: (%s; %s), mean: %s" % (fscore_method, lower, upper, mean))
        ret[fscore_method] = "Coarse-seq F-scores (%s) interval: (%s; %s), mean: %s" % (fscore_method, lower, upper, mean)
    return ret


def run():
    log.info('Loading datasets...')
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = read_split_data(ARGS['tag'],
                                                                                                ARGS['dir'],
                                                                                                ARGS['split'],
                                                                                                oversample_train="half")
    _, coarse_labels = read_coarse_test_data(ARGS['tag'], ARGS['dir'], ARGS['split'])
    # train_texts = train_texts[:2000]
    # train_labels = train_labels[:2000]
    # dev_texts = dev_texts[:500]
    # dev_labels = dev_labels[:500]
    # test_texts = test_texts[:2000]
    # test_labels = test_labels[:2000]


    log.info('Loading tokenizer and tokenizing...')
    # model_pth = Path('results', ARGS['run_id'], ARGS['split'], 'checkpoint-2500')
    # tokenizer = AutoTokenizer.from_pretrained(model_pth, use_fast=False, truncation_side='left')
    tokenizer = AutoTokenizer.from_pretrained(ARGS['model_name'], use_fast=False, truncation_side='left')
    assert tokenizer.truncation_side == 'left'
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=ARGS['max_length'])
    dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, max_length=ARGS['max_length'])
    train_dataset = EncodingDataset(train_encodings, train_labels)
    dev_dataset = EncodingDataset(dev_encodings, dev_labels)

    log.info('Load model and move to GPU...')
    if ARGS['model_name'] == 'Seznam/small-e-czech':
        model = ElectraForSequenceClassification.from_pretrained(
           ARGS['model_name'], num_labels=6, problem_type="multi_label_classification").to("cuda")
        # model = ElectraForSequenceClassification.from_pretrained(
        #      model_pth, num_labels=6, problem_type="multi_label_classification").to("cuda")
    elif ARGS['model_name'] == 'ufal/robeczech-base':
        model = RobertaForSequenceClassification.from_pretrained(
            ARGS['model_name'], num_labels=6, problem_type="multi_label_classification").to("cuda")
    elif ARGS['model_name'] == 'xlm-roberta-base' or ARGS['model_name'] == 'xlm-roberta-large':
        model = XLMRobertaForSequenceClassification.from_pretrained(
            ARGS['model_name'], num_labels=6, problem_type="multi_label_classification").to("cuda")
    else:
        raise Exception('Unknown model name.')

    log.info('Set training ARGS...')
    training_args = TrainingArguments(
        output_dir=ARGS['output_dir'],  # output directory
        num_train_epochs=ARGS['epochs'],  # total number of training epochs
        per_device_train_batch_size=ARGS['per_device_batch'],  # batch size per device during training
        per_device_eval_batch_size=ARGS['per_device_batch'],  # batch size for evaluation
        gradient_accumulation_steps=ARGS['gradient_acc'],
        warmup_steps=ARGS['w_steps'],  # number of warmup steps for learning rate scheduler
        # weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        metric_for_best_model='f1',
        logging_strategy=ARGS['logging_strategy'],
        evaluation_strategy=ARGS['evaluation_strategy'],
        save_strategy=ARGS['save_strategy'],
        learning_rate=ARGS['learning_rate'],
        save_total_limit=ARGS['save_total_limit'],
        max_steps=ARGS['max_steps'],
        eval_steps=ARGS['eval_steps'],
        logging_steps=ARGS['logging_steps'],
        save_steps=ARGS['save_steps'],
        disable_tqdm=False,
        fp16=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[LogCallback, EarlyStoppingCallback(early_stopping_patience=ARGS['early_stopping_patience'])]
    )

    log.info('Train...')
    trainer.train()
    # log.info(trainer.evaluate())
    model_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    trainer.save_model(model_pth)
    tokenizer.save_pretrained(model_pth)

    log.info('Evaluation multi-multi...')
    preds = [predict_one_ml(tt, tokenizer, model) for tt in test_texts]
    results_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}_m2m.json'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    # roc_auc = roc_auc_score(test_labels, preds, average='weighted')
    # ci: Dict = bootstrap_michal(test_labels, preds)
    cr = classification_report(test_labels, preds)
    log.info(cr)
    cfm = list(multilabel_confusion_matrix(test_labels, preds))
    with open(results_pth, 'w', encoding='utf-8') as outfile:
        ARGS['output_dir'] = str(ARGS['output_dir'])  # need to convert Path to str before dump
        result = {
            'micro_f1': str(multi_f1(test_labels, preds, method="micro")),
            'macro_f1': str(multi_f1(test_labels, preds, method="macro")),
            'weighted_f1': str(multi_f1(test_labels, preds, method="weighted")),
            'test_cfm': str(cfm),
            'cr': str(cr),
            'args': ARGS
        }
        result['args']['class_ratio'] = result['args']['class_ratio'].tolist()
        # result.update(ci)
        json.dump(result, outfile, ensure_ascii=False)

    # log.info('Evaluation multi-coarse...')
    # preds = [predict_one_m2b(tt, tokenizer, model) for tt in test_texts]
    # results_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}_m2c.json'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    # bf1 = bin_f1(coarse_labels, preds)
    # roc_auc = roc_auc_score(coarse_labels, preds, average='weighted')
    # ci: Dict = bootstrap_michal(coarse_labels, preds)
    # cfm = list(confusion_matrix(coarse_labels, preds).tolist())
    # with open(results_pth, 'w', encoding='utf-8') as outfile:
    #     ARGS['output_dir'] = str(ARGS['output_dir'])  # need to convert Path to str before dump
    #     result = {
    #         'test_f1': bf1,
    #         'auc': f'{roc_auc:.4f}',
    #         'cfm': str(cfm)
    #     }
    #     result.update(ci)
    #     json.dump(result, outfile, ensure_ascii=False)
    #
    # rminsidedir(ARGS['output_dir'], 'checkpoint')
    # log.info('Finished.')
    # return model, tokenizer


def main():
    # set seed and GPUs
    set_seed(1)
    no_gpus = set_gpus(ARGS['gpu'])
    ARGS['visible_devices'] = no_gpus

    # run
    model, tokenizer = run()  # results are saved in split_# dir
    plot_loss()  # only plots for this split
    return model, tokenizer


if __name__ == '__main__':
    # load args
    ARGS = load_config(Path(sys.argv[1]))
    ARGS['split'] = str(sys.argv[2])  # split_#
    ARGS['gpu'] = str(sys.argv[3])  # single string number
    ARGS['output_dir'] = mk_out_dir(ARGS['split'])  # results/<run_id>/<split_#>

    # create global loss logger
    LOSS = []
    F1 = []
    CFM = []

    # file logging
    log_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}.log'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    file_handler = log.FileHandler(filename=log_pth)
    std_handler = log.StreamHandler(sys.stdout)
    err_handler = log.StreamHandler(sys.stderr)
    handlers = [file_handler, std_handler, err_handler]
    log.basicConfig(
        level=log.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )
    wandb.init(project="irtisdb")
    # wandb.init(mode="disabled")
    # run
    try:
        MODEL, TOKENIZER = main()
    except Exception as e:
        log.exception(e)
        raise

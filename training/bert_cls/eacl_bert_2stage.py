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
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, EvalPrediction

from bert_cls.utils import load_config, mk_out_dir, set_seed, set_gpus, rminsidedir, load_model_bin, \
    oversample_multiclass_equal


def transform_and_filter(train_texts, train_labels):
    """Filters out 0 labels and multi-label anntations and cuts off the 0 label"""
    tt = []
    tl = []
    for t,label in zip(train_texts, train_labels):
        if label[0] != 1 and sum(label[1:]) == 1:
            tt.append(t)
            tl.append(label[1:])
    return tt, tl


def read_split_data(tag: str, dirr: Path, split_i: str, oversample_train="none"):
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.train')) as json_data:
        root = json.load(json_data)
        train_texts = [x[0] for x in root['multi']]
        train_labels = [x[1] for x in root['multi']]
        train_texts, train_labels = transform_and_filter(train_texts, train_labels)
        if oversample_train == "none":
            pass
        elif oversample_train == "equal":
            train_texts, train_labels, counts = oversample_multiclass_equal(train_texts, train_labels)
            ARGS['class_ratio'] = counts
        else:
            raise NotImplemented
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.dev')) as json_data:
        root = json.load(json_data)
        dev_texts = [x[0] for x in root['multi']]
        dev_labels = [x[1] for x in root['multi']]
        dev_texts, dev_labels = transform_and_filter(dev_texts, dev_labels)
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.test')) as json_data:
        root = json.load(json_data)
        test_texts = [x[0] for x in root['multi']]
        test_labels = [x[1] for x in root['multi']]
        test_texts, test_labels = transform_and_filter(test_texts, test_labels)
    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


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
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_macro_average,
               'f1_micro': f1_micro_average,
               'f1_weighted': f1_weighted_average,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


def multi_f1(true_labels, pred_labels, method):
    return round(f1_score(y_true=true_labels, y_pred=pred_labels, average=method), 4)


def get_preds(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=ARGS['max_length'],
                       return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    preds = torch.argmax(probs, -1).detach().cpu()
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs.detach().cpu() >= 0.5)] = 1
    return predictions


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


def run():
    log.info('Loading datasets...')
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = read_split_data(ARGS['tag'],
                                                                                                ARGS['dir'],
                                                                                                ARGS['split'],
                                                                                                oversample_train="equal")
    # train_texts = train_texts[:2000]
    # train_labels = train_labels[:2000]
    # dev_texts = dev_texts[:500]
    # dev_labels = dev_labels[:500]
    # test_texts = test_texts[:2000]
    # test_labels = test_labels[:2000]


    log.info('Tokenizing...')
    # model_pth = Path('results', ARGS['run_id'], ARGS['split'], 'checkpoint-2500')
    # tokenizer = AutoTokenizer.from_pretrained(model_pth, use_fast=False, truncation_side='left')
    tokenizer = AutoTokenizer.from_pretrained(ARGS['model_name'], use_fast=False, truncation_side='left')
    assert tokenizer.truncation_side == 'left'
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=ARGS['max_length'])
    dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, max_length=ARGS['max_length'])
    train_dataset = EncodingDataset(train_encodings, train_labels)
    dev_dataset = EncodingDataset(dev_encodings, dev_labels)

    log.info('Load model and move to GPU...')
    model = load_model_bin(ARGS['model_name'], ARGS['target_names'])

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
    model_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    trainer.save_model(model_pth)
    tokenizer.save_pretrained(model_pth)

    log.info('Evaluation...')
    preds = []
    test_batch_size = 64
    for batch_offset in tqdm(range(0, len(test_labels), test_batch_size)):
        input_text_batch = [text for text in test_texts[batch_offset:batch_offset + test_batch_size]]
        decoded = get_preds(input_text_batch, tokenizer, model)
        preds.extend(decoded)

    cr = classification_report(test_labels, preds)
    log.info(cr)
    cfm = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(preds, axis=1)).tolist()
    results_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}.json'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
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
        json.dump(result, outfile, ensure_ascii=False)
    rminsidedir(ARGS['output_dir'], 'checkpoint')
    log.info('Finished.')
    return model, tokenizer


def main():
    # set seed and GPUs
    set_seed(1)
    no_gpus = set_gpus(ARGS['gpu'])
    ARGS['visible_devices'] = no_gpus

    # run
    model, tokenizer = run()  # results are saved in split_# dir
    # plot_loss()  # only plots for this split
    return model, tokenizer


if __name__ == '__main__':
    # load args
    ARGS = load_config(Path(sys.argv[1]))
    ARGS['split'] = str(sys.argv[2])  # split_#
    ARGS['gpu'] = str(sys.argv[3])  # single string number
    ARGS['output_dir'] = mk_out_dir(ARGS['run_id'], ARGS['split'])  # results/<run_id>/<split_#>

    # create global loss logger
    LOSS = []
    F1 = []
    CFM = []

    # file logging
    log_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}.log'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    file_handler = log.FileHandler(filename=log_pth)
    std_handler = log.StreamHandler(sys.stdout)
    handlers = [file_handler, std_handler]
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

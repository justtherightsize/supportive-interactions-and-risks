import json
import sys
from pathlib import Path
import logging as log
import torch
import transformers
import wandb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    multilabel_confusion_matrix, classification_report
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback

from bert_cls.utils import load_config, mk_out_dir, set_seed, set_gpus, rminsidedir, load_model_bin, \
    oversample_coarse_half


def read_split_data(tag: str, dirr: Path, split_i: str, oversample_train="none"):
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.train')) as json_data:
        root = json.load(json_data)
        train_texts = [x[0] for x in root['coarse']]
        train_labels = [x[1] for x in root['coarse']]
        if oversample_train == "none":
            pass
        elif oversample_train == "half":
            train_texts, train_labels, clr = oversample_coarse_half(train_texts, train_labels)
            ARGS['class_ratio'] = clr
        else:
            raise NotImplemented
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.dev')) as json_data:
        root = json.load(json_data)
        dev_texts = [x[0] for x in root['coarse']]
        dev_labels = [x[1] for x in root['coarse']]
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.test')) as json_data:
        root = json.load(json_data)
        test_texts = [x[0] for x in root['coarse']]
        test_labels = [x[1] for x in root['coarse']]
    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


def compute_metrics(pred):
    labels = pred.label_ids
    pred = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    cfm = confusion_matrix(labels, pred).tolist()
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "cfm": cfm}


def get_preds(text, tokenizer, model):
    """
    Return the predicted classes
    """
    inputs = tokenizer(text, padding=True, truncation=True, max_length=ARGS['max_length'],
                       return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    preds = torch.argmax(probs, -1).detach().cpu()
    return preds


class EncodingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


class LogCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if 'loss' in state.log_history[-2]:
            LOSS.append(state.log_history[-2]['loss'])
        if 'eval_f1' in state.log_history[-1]:
            F1.append(state.log_history[-1]['eval_f1'])


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(ARGS['class_ratio']).to("cuda"))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def run():
    log.info('Loading datasets...')
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = read_split_data(ARGS['tag'],
                                                                                                ARGS['dir'],
                                                                                                ARGS['split'],
                                                                                                oversample_train="half")
    # train_texts = train_texts[:1000]
    # train_labels = train_labels[:1000]
    # dev_texts = dev_texts[:100]
    # dev_labels = dev_labels[:100]
    # test_texts = test_texts[:500]
    # test_labels = test_labels[:500]


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
    trainer = CustomTrainer(
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
    test_batch_size = 16
    for batch_offset in tqdm(range(0, len(test_labels), test_batch_size)):
        input_text_batch = [text for text in test_texts[batch_offset:batch_offset + test_batch_size]]
        decoded = get_preds(input_text_batch, tokenizer, model)
        preds.extend(decoded)
    cr = classification_report(test_labels, preds, output_dict=True)
    log.info(cr)
    cfm = list(confusion_matrix(test_labels, preds))
    results_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}.json'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    with open(results_pth, 'w', encoding='utf-8') as outfile:
        ARGS['output_dir'] = str(ARGS['output_dir'])  # need to convert Path to str before dump
        result = {
            'test_cfm': str(cfm),
            'cr': str(cr),
            'args': ARGS
        }
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

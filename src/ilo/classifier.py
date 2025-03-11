import argparse
import os
import sys
sys.path.append('classifier/ilo')
import logging
import pickle
import random
import json
import time
import re
from itertools import permutations
from functools import partial
from collections import Counter
from torch.utils.data import Dataset

utils = os.path.abspath('../src/utils/') # Relative path to utils scripts
sys.path.append(utils)

from evaluation import createResults, convertLabels
from preprocessing import loadDataset, CATEGORY_MAPPINGS, POLARITY_MAPPINGS_POL_TO_TERM, POLARITY_MAPPINGS_TERM_TO_POL, TEXT_TEMPLATES, TEXT_PATTERNS, IT_TOKENS
import numpy as np
import torch
import pandas as pd

from transformers.models.t5.modeling_t5 import *
from transformers import AdamW, T5Tokenizer
from t5_score import MyT5ForConditionalGenerationScore
from t5 import MyT5ForConditionalGeneration

from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl

from tqdm import tqdm

LABEL_SPACE = ['ambience general:POSITIVE', 'ambience general:NEUTRAL', 'ambience general:NEGATIVE', 'drinks prices:POSITIVE', 'drinks prices:NEUTRAL', 'drinks prices:NEGATIVE', 'drinks quality:POSITIVE', 'drinks quality:NEUTRAL', 'drinks quality:NEGATIVE', 'drinks style_options:POSITIVE', 'drinks style_options:NEUTRAL', 'drinks style_options:NEGATIVE', 'food prices:POSITIVE', 'food prices:NEUTRAL', 'food prices:NEGATIVE', 'food quality:POSITIVE', 'food quality:NEUTRAL', 'food quality:NEGATIVE', 'food style_options:POSITIVE', 'food style_options:NEUTRAL', 'food style_options:NEGATIVE', 'location general:POSITIVE', 'location general:NEUTRAL', 'location general:NEGATIVE', 'restaurant general:POSITIVE', 'restaurant general:NEUTRAL', 'restaurant general:NEGATIVE', 'restaurant miscellaneous:POSITIVE', 'restaurant miscellaneous:NEUTRAL', 'restaurant miscellaneous:NEGATIVE', 'restaurant prices:POSITIVE', 'restaurant prices:NEUTRAL', 'restaurant prices:NEGATIVE', 'service general:POSITIVE', 'service general:NEUTRAL', 'service general:NEGATIVE']

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
class ABSADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 dataset,
                 lang,
                 data_type,
                 top_k,
                 args,
                 max_len=128):
            
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.lang = lang
        self.dataset = dataset
        self.args = args

        self.top_k = top_k

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

    def _build_examples(self):
        
        inputs, targets = get_transformed_io(self.dataset,
                                             self.data_type, self.top_k,
                                             self.args)

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")
            
            # for ACOS Restaurant and Laptop dataset
            # the max target length is much longer than 200
            # we need to set a larger max length for inference
            target_max_length = 1024 if self.data_type == "test" else self.max_len

            tokenized_target = self.tokenizer.batch_encode_plus(
                [target],
                max_length=target_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

def read_line_examples_from_df(df,
                                 lowercase,
                                 silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    tasks, datas = [], []
    sents, labels = [], []
    
    for index, row in df.iterrows():
        if lowercase:
            sents.append(formatText(row['text'].lower()).split())
            labels.append([(label[2].lower(), label[0], label[1]) for label in row['labels']])
        else:
            sents.append(formatText(row['text']).split())
            labels.append([(label[2], label[0], label[1]) for label in row['labels']])

    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels

def get_transformed_io(dataset, data_type, top_k, args):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_df(
        dataset, args.lowercase)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]
    if data_type == "train":
        new_inputs, targets = get_para_tasd_targets(inputs, labels, top_k, args, args.lang)
    else:
        targets = get_para_tasd_targets_test(inputs, labels, args.lang)
        return inputs, targets
    print(len(inputs), len(new_inputs), len(targets))
    return new_inputs, targets

def get_para_tasd_targets(sents, labels, top_k, args, lang):
    """
    Erzeugt Zielsequenzen für den TASD-Task im Paraphrase-Paradigma.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = MyT5ForConditionalGenerationScore.from_pretrained(args.model_name_or_path).to(device)

    targets = []
    new_sents = []
    data_count = {}

    for i in range(len(sents)):
        label = labels[i]
        cur_sent = sents[i]

        if len(label) in data_count:
            data_count[len(label)] += 1
        else:
            data_count[len(label)] = 1

        quad_list = []
        for quad in label:
            at, ac, sp = quad

            ## Sprachspezifisch
            man_sp =  POLARITY_MAPPINGS_POL_TO_TERM[lang][sp]  # z.B. 'POS' -> 'good'

            if at == "NULL":  # Für implizite Aspekte
                at = IT_TOKENS[lang]

            quad = [f"[AT] {at}", f"[AC] {ac}", f"[SP] {man_sp}"]
            x = permutations(quad)
            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[:4])  # Extrahiert das Tag, z.B. "[AT]"
                    content.append(e[4:])  # Extrahiert den Text
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            quad_list.append(permute_object)

        best_order = choose_best_order_batch(quad_list, cur_sent, model, tokenizer, device, top_k, task="tasd")

        for b_o in best_order:
            targets.append(b_o)
            new_sents.append(cur_sent)

    return new_sents, targets

def get_para_tasd_targets_test(sents, labels, lang):
    """
    Erzeugt die Ziel-Sätze für den Testmodus im TASD-Task.
    """
    targets = []
    for label in labels:
        all_triplet_sentences = []
        for triplet in label:
            at, ac, sp = triplet

            ## Sprachspezifisch
            if at == 'NULL':  # Für implizite Aspekte
                at = globals()[f'IT_TOKEN_{lang.upper()}']

            # Triplet ohne OT (Opinion Term)
            triplet_list = [f"[AT] {at}", f"[AC] {ac}", f"[SP] {sp}"]
            one_triplet_sentence = " ".join(triplet_list)
            all_triplet_sentences.append(one_triplet_sentence)

        target = ' [SSEP] '.join(all_triplet_sentences)
        targets.append(target)
    return targets


def order_scores_function(quad_list, cur_sent, model, tokenizer, device, task):
    """
    Berechnet die Scores für verschiedene Reihenfolgen von Quadruples basierend auf dem Task (tasd oder asqp).
    
    Args:
        quad_list: Liste von Quadruple-Daten mit möglichen Reihenfolgen.
        cur_sent: Der aktuelle Satz (Eingabe).
        model: Das Modell zur Berechnung der Scores.
        tokenizer: Tokenizer zum Verarbeiten der Eingaben und Ziele.
        device: Zielgerät (z. B. GPU oder CPU).
        task: Der spezifische Task, entweder 'tasd' oder 'asqp'.
    
    Returns:
        results: Ein Dictionary mit Scores für jede Reihenfolge.
    """
    # Definiere mögliche Reihenfolgen basierend auf dem Task

    q = ["[AT]", "[AC]", "[SP]"]  # Keine OT-Komponente in TASD
    
    all_orders = permutations(q)
    all_orders_list = []

    all_targets = []
    all_inputs = []
    cur_sent = " ".join(cur_sent)  # Satz als String zusammenfügen

    for each_order in all_orders:
        cur_order = " ".join(each_order)
        all_orders_list.append(cur_order)
        cur_target = []
        for each_q in quad_list:
            cur_target.append(each_q[cur_order][0])  # Extrahiere die entsprechende Zielsequenz

        all_inputs.append(cur_sent)
        all_targets.append(" ".join(cur_target))

    # Tokenisiere Eingaben und Ziele
    tokenized_input = tokenizer.batch_encode_plus(
        all_inputs, max_length=200, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    tokenized_target = tokenizer.batch_encode_plus(
        all_targets, max_length=200, padding="max_length",
        truncation=True, return_tensors="pt"
    )

    target_ids = tokenized_target["input_ids"].to(device)

    # Maskiere Padding-Tokens
    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100

    # Modell-Ausgabe
    outputs = model(
        input_ids=tokenized_input["input_ids"].to(device),
        attention_mask=tokenized_input["attention_mask"].to(device),
        labels=target_ids,
        decoder_attention_mask=tokenized_target["attention_mask"].to(device)
    )

    loss, entropy = outputs[0]
    results = {}

    # Ergebnisse speichern
    for i, _ in enumerate(all_orders_list):
        cur_order = all_orders_list[i]
        results[cur_order] = {"loss": loss[i], "entropy": entropy[i]}

    return results


def choose_best_order_batch(quad_list, cur_sent, model, tokenizer, device, top_k, task):
    """
    Wählt die beste Reihenfolge der Quads basierend auf dem Task (tasd oder asqp).
    """
    q = ["[AT]", "[AC]", "[SP]"]
    
    all_orders = permutations(q)
    all_orders_list = []

    all_targets = []
    all_inputs = []
    cur_sent = " ".join(cur_sent)
    for each_order in all_orders:
        cur_order = " ".join(each_order)
        all_orders_list.append(cur_order)
        cur_target = []
        for each_q in quad_list:
            cur_target.append(each_q[cur_order][0])

        all_inputs.append(cur_sent)
        all_targets.append(" ".join(cur_target))

    tokenized_input = tokenizer.batch_encode_plus(
        all_inputs, max_length=200, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    tokenized_target = tokenizer.batch_encode_plus(
        all_targets, max_length=200, padding="max_length",
        truncation=True, return_tensors="pt"
    )

    target_ids = tokenized_target["input_ids"].to(device)
    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
    outputs = model(
        input_ids=tokenized_input["input_ids"].to(device),
        attention_mask=tokenized_input["attention_mask"].to(device),
        labels=target_ids,
        decoder_attention_mask=tokenized_target["attention_mask"].to(device)
    )

    loss, entropy = outputs[0]

    indexes = np.argsort(np.array(entropy))  # Sortierung nach Entropie
    all_chosen = []
    choosen_indexes = indexes[0: top_k]
    for e in choosen_indexes:
        cur_order = all_orders_list[e]
        best_quad = []
        for each_q in quad_list:
            best_quad.append(each_q[cur_order][1])
        best_quad = " [SSEP] ".join(best_quad)
        all_chosen.append(best_quad)
    
    return all_chosen

    
def formatText(text):
    text = re.sub(r'([(".,!?;:/)])', r" \1", text)
    text = re.sub(r'(["„“…])', r'', text)
    text = re.sub(r'([\'])', r' \1', text)
    # text = re.sub(r'([-])', r' \1 ', text)
    text = re.sub(r'([\s\s])', r' ', text)
    text = re.sub(r"\b(I|You|We|They|He|She|It|Don|Didn|Doesn|Can|Couldn|Wouldn|Shouldn|Won|Would|Wasn|Aren|Ain|Isn|Hasn|Haven|Weren|Mightn|Mustn)('|’)(m|t|ll|ve|re|s|d)\b", r"\1 \2\3", text)
    return re.sub(r"\s+", " ", text).strip()

def splitForEvalSetting(dataset, eval_type):
        """Handles dataset splitting and cross-validation settings."""
        train, test, label_space = dataset
        split = eval_type.split('_')[1] if '_' in eval_type else False
        
        if split:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            train_idx, val_idx = list(kf.split(train, None))[int(split)]
            train, test = train.iloc[train_idx], train.iloc[val_idx]
            print(f"Creating CV splits; using split {split} with random_state 42")

        print(f"Train set size: {len(train)}, Test set size: {len(test)}")
        return train, test, label_space

class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, hparams, tfm_model, tokenizer, train_dataset):
        super(T5FineTuner, self).__init__()
        self.hparams.update(vars(hparams))
        # hparams = vars(hparams)
        # for key in hparams.keys():
        #     try:
        #         self.hparams[key]=hparams[key]
        #     except:
        #         print("Param not found: ", key)
        self.model = tfm_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        adjusted_batch = int(self.hparams.train_batch_size/self.hparams.gradient_accumulation_steps) # CLI batch scales with gradient steps
        
        dataloader = DataLoader(self.train_dataset, batch_size=adjusted_batch,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    print(f"f1-score: {f1}")
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores

def extract_spans_para(task, seq, seq_type, lang):
    """
    Extrahiert die Komponenten aus der Zielsequenz basierend auf dem Task (tasd oder asqp).

    Args:
        task: Der spezifische Task, entweder 'tasd' oder 'asqp'.
        seq: Die Sequenz, die analysiert werden soll.
        seq_type: Typ der Sequenz (z. B. train, test).

    Returns:
        quads: Eine Liste von Triplets oder Quadruples abhängig vom Task.
    """
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]

    for s in sents:
        try:
            # Indexe für die Komponenten finden
            index_ac = s.index("[AC]")
            index_sp = s.index("[SP]")
            index_at = s.index("[AT]")

            combined_list = [index_ac, index_sp, index_at]
            arg_index_list = list(np.argsort(combined_list))  # Reihenfolge sortieren

            result = []
            for i in range(len(combined_list)):
                start = combined_list[i] + 4
                sort_index = arg_index_list.index(i)
                if sort_index < 2:  # Nur die nächsten zwei Elemente überprüfen
                    next_ = arg_index_list[sort_index + 1]
                    re = s[start: combined_list[next_]]
                else:
                    re = s[start:]
                result.append(re.strip())

            ac, sp, at = result

            # Wenn der Aspekt-Text implizit ist
            if at.lower() == IT_TOKENS[lang]:
                at = 'NULL'
        except ValueError:
            try:
                # Fehlerhafte Sequenzen ignorieren
                pass
            except UnicodeEncodeError:
                pass
            ac, at, sp = '', '', ''

        quads.append((ac, at, sp))  # Triplet für TASD speichern
    return quads


def compute_scores(pred_seqs, gold_seqs, task, lang):
    """
    Compute model performance by extracting predicted and gold spans,
    formatting them, and calculating evaluation metrics.
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels = [extract_spans_para(task, gold_seq, 'gold', lang) for gold_seq in gold_seqs]
    all_preds = [extract_spans_para(task, pred_seq, 'pred', lang) for pred_seq in pred_seqs]
        
    print(all_preds[:5])
    print(all_labels[:5])
    
    try:
        preds = [
            [f"{lbl[0]}:{POLARITY_MAPPINGS_TERM_TO_POL[lang].get(lbl[2], '').upper()}:{lbl[1]}"
             for lbl in pred if f"{lbl[0]}::{lbl[1]}" != "::"]
            for pred in all_preds
        ]

        golds = [
            [f"{lbl[0]}:{lbl[2].upper()}:{lbl[1]}" for lbl in gold if f"{lbl[0]}::{lbl[1]}" != "::"]
            for gold in all_labels
        ]
            
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Error encountered in processing predictions or labels.")
        print("Sample all_preds:", all_preds[:5])
        print("Sample all_labels:", all_labels[:5])
        return None  # Early exit if KeyError occurs

    print(preds[:5])
    print(golds[:5])

    scores_dfs = createResults(preds, golds, LABEL_SPACE, task)
    
    scores = compute_f1_scores(all_preds, all_labels)
    print('ILO F1-Micro: ', scores['f1'])
    
    return scores_dfs, all_labels, pred_seqs


def evaluate(data_loader, model, tokenizer, args):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device('cuda:0')
    model.model.to(device)

    model.model.eval()

    outputs, targets = [], []

    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.model.generate(input_ids=batch['source_ids'].to(device),
                                    attention_mask=batch['source_mask'].to(device),
                                    max_length=args.max_seq_length,
                                    num_beams=5)  # num_beams=5 num_beams=8, early_stopping=True)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)

    scores, all_labels, preds = compute_scores(outputs, targets, args.task, args.lang)

    return scores, preds


def train_function_ilo(args):
    set_seed(args.seed)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    train, test, _ = splitForEvalSetting(loadDataset(args.data_path, args.lang, args.data_setting), args.eval_type)
    
    args.dataset = f'rest-{args.lang}'
    args.lang = 'en' if args.lang_setting != 'adapted' else args.lang
    
    train_dataset = ABSADataset(tokenizer=tokenizer,
                              dataset=train,
                              lang=args.lang,
                              data_type='train',
                              top_k=args.top_k,
                              args=args,
                              max_len=args.max_seq_length)
    
    test_dataset = ABSADataset(tokenizer=tokenizer,
                              dataset=test,
                              lang=args.lang,
                              data_type=args.eval_type,
                              top_k=args.top_k,
                              args=args,
                              max_len=args.max_seq_length)
    
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4)
    
    
    # training process
    if args.do_train:

        # initialize the T5 model
        tfm_model = MyT5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model = T5FineTuner(args, tfm_model, tokenizer, train_dataset)
        
        # prepare for trainer
        if torch.cuda.is_available():
            gpus = 1
        else:
            gpus = None
        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=gpus,  # args.n_gpu,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            check_val_every_n_epoch=5,
            logger=False,
        )

        trainer = pl.Trainer(**train_params)

        try:
            trainer.fit(model)
        except KeyboardInterrupt:
            print("Training has been stopped manually.")
            
        # save the final model
        # model.model.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)
        
        results = evaluate(test_loader, model, tokenizer, args)
        return results

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--data_path", default="../data/", type=str)
    parser.add_argument("--task", default='asqp', type=str,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='rest15', type=str,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", default=False, action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_inference", default=True,
                        help="Whether to run inference with trained checkpoints")
    parser.add_argument("--output_dir",
                        default='outputs/temp',
                        type=str,
                        help="Output directory")
    
    parser.add_argument(
        "--lang",
        default='en',
        choices=["de", "en", "nl", "ru", "cs", "fr", "es", "tr"],
        type=str,
        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument(
        "--eval_type",
        default='test',
        choices=["test", "dev"],
        type=str,
    )
    parser.add_argument(
        "--data_setting",
        default="orig",
        choices=["orig", "balanced", "multi_id", "multi_od"],
        type=str,
    )
    parser.add_argument(
        "--lang_setting",
        default="full",
        choices=["orig", "adapted"],
        type=str,
    )
    parser.add_argument("--lowercase", action='store_true')
    parser.add_argument('--seed',
                        type=int,
                        default=25,
                        help="random seed for initialization")
    
    # other parameters
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--top_k", default=1, type=int)
    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = init_args()
    set_seed(args.seed)
    
    if 'multi' in args.data_setting:
        (f1_balanced, preds_balanced), (f1_orig, preds_orig) = train_function_ilo(args)
    # f1_str = "F1: {}".format(f1_res['f1'])
        output_path = os.path.join(args.output_dir, f'{args.task}_{args.lang}_{args.lang_setting}_{args.eval_type}_{args.data_setting.replace("_","-")}-b_{args.learning_rate}_{args.train_batch_size}_{args.num_train_epochs}_{args.seed}')
        os.makedirs(output_path, exist_ok=True)
    
        pd.DataFrame.from_dict(f1_balanced[0]).transpose().to_csv(output_path + '/metrics_asp.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_balanced[1]).transpose().to_csv(output_path + '/metrics_asp_pol.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_balanced[2]).transpose().to_csv(output_path + '/metrics_pairs.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_balanced[3]).transpose().to_csv(output_path + '/metrics_pol.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_balanced[4]).transpose().to_csv(output_path + '/metrics_phrases.tsv', sep = "\t")

        with open(os.path.join(output_path, 'predictions.json'), "w", encoding="utf-8") as f:
            json.dump({'predictions': preds_balanced}, f, indent=4, ensure_ascii=False)

        output_path = os.path.join(args.output_dir, f'{args.task}_{args.lang}_{args.lang_setting}_{args.eval_type}_{args.data_setting.replace("_","-")}-o_{args.learning_rate}_{args.train_batch_size}_{args.num_train_epochs}_{args.seed}')
        os.makedirs(output_path, exist_ok=True)
    
        pd.DataFrame.from_dict(f1_orig[0]).transpose().to_csv(output_path + '/metrics_asp.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_orig[1]).transpose().to_csv(output_path + '/metrics_asp_pol.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_orig[2]).transpose().to_csv(output_path + '/metrics_pairs.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_orig[3]).transpose().to_csv(output_path + '/metrics_pol.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_orig[4]).transpose().to_csv(output_path + '/metrics_phrases.tsv', sep = "\t")

        with open(os.path.join(output_path, 'predictions.json'), "w", encoding="utf-8") as f:
                json.dump({'predictions': preds_orig}, f, indent=4, ensure_ascii=False)
    else:
        f1_res, preds = train_function_ilo(args)
    
        # f1_str = "F1: {}".format(f1_res['f1'])
        output_path = os.path.join(args.output_dir, f'{args.task}_{args.lang}_{args.lang_setting}_{args.eval_type}_{args.data_setting}_{args.learning_rate}_{args.train_batch_size}_{args.num_train_epochs}_{args.seed}')
        os.makedirs(output_path, exist_ok=True)
    
        pd.DataFrame.from_dict(f1_res[0]).transpose().to_csv(output_path + '/metrics_asp.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_res[1]).transpose().to_csv(output_path + '/metrics_asp_pol.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_res[2]).transpose().to_csv(output_path + '/metrics_pairs.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_res[3]).transpose().to_csv(output_path + '/metrics_pol.tsv', sep = "\t")
        pd.DataFrame.from_dict(f1_res[4]).transpose().to_csv(output_path + '/metrics_phrases.tsv', sep = "\t")

        with open(os.path.join(output_path, 'predictions.json'), "w", encoding="utf-8") as f:
            json.dump({'predictions': preds}, f, indent=4, ensure_ascii=False)
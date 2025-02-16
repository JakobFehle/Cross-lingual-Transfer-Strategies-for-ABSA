import pandas as pd
import numpy as np
import torch
import os, sys
import transformers
import argparse
import re

utils = os.path.abspath('../src/utils/') # Relative path to utils scripts
sys.path.append(utils)

from config import Config
from dataclasses import dataclass, field
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorForSeq2Seq
from preprocessing import loadDataset
from evaluation import createResults, convertLabels, extractAspects
from datetime import datetime, timedelta
from typing import Optional

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")
warnings.filterwarnings("ignore", category=UserWarning)

DATASETS = ['nl', 'de', 'en', 'cs', 'ru', 'fr', 'es']

CAT_TO_TERM_DE = {
    "SERVICE": "Service",
    "FOOD": "Essen",
    "PRICE": "Preis",
    "AMBIENCE": "Ambiente",
    "GENERAL-IMPRESSION": "Allgemeiner Eindruck"
}

CAT_TO_TERM_EN = {
    "ambience general": "Ambience general",
    "drinks prices": "Drinks prices",
    "drinks quality": "Drinks quality",
    "Drinks style options": "Drinks style options",
    "FOOD#PRICES": "Food prices",
    "FOOD#QUALITY": "Food quality",
    "FOOD#STYLE_OPTIONS": "Food style options",
    "LOCATION#GENERAL": "Location general",
    "RESTAURANT#GENERAL": "Restaurant general",
    "RESTAURANT#MISCELLANEOUS": "Restaurant miscellaneous",
    "RESTAURANT#PRICES": "Restaurant prices",
    "SERVICE#GENERAL": "Service general"
}

POL_TO_TERM_DE = {
    "negative": "schlecht",
    "neutral": "ok",
    "positive": "gut"
}

POL_TO_TERM_EN = {
    "negative": "bad",
    "neutral": "ok",
    "positive": "great"
}

TERM_TO_CAT_DE = {
    "Service": "SERVICE",
    "Essen": "FOOD",
    "Preis": "PRICE",
    "Ambiente": "AMBIENCE",
    "Allgemeiner Eindruck": "GENERAL-IMPRESSION"
}

TERM_TO_CAT_EN = {
    "Ambience general": "AMBIENCE#GENERAL",
    "Drinks prices": "DRINKS#PRICES",
    "Drinks quality": "DRINKS#QUALITY",
    "Drinks style options": "DRINKS#STYLE_OPTIONS",
    "Food prices": "FOOD#PRICES",
    "Food quality": "FOOD#QUALITY",
    "Food style options": "FOOD#STYLE_OPTIONS",
    "Location general": "LOCATION#GENERAL",
    "Restaurant general": "RESTAURANT#GENERAL",
    "Restaurant miscellaneous": "RESTAURANT#MISCELLANEOUS",
    "Restaurant prices": "RESTAURANT#PRICES",
    "Service general": "SERVICE#GENERAL"
}

TERM_TO_POL_GER = {
    "schlecht": "negative",
    "ok": "neutral",
    "gut": "positive"
}

TERM_TO_POL_EN = {
    "bad": "negative",
    "ok": "neutral",
    "great": "positive"
}

TEXT_TEMPLATE_DE = "{ac_text} ist {polarity_text}, weil {aspect_term_text} {polarity_text} ist."
TEXT_TEMPLATE_EN = "{ac_text} is {polarity_text}, because {aspect_term_text} is {polarity_text}."

TEXT_PATTERN_DE = r"(.*) ist (.*), weil (.*) (.*) ist."
TEXT_PATTERN_EN = r"(.*) is (.*), because (.*) is (.*)."

IT_TOKEN_DE = 'es'
IT_TOKEN_EN = 'it'

class CustomDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

class ParaphraseABSA:
    def __init__(self, args):
        self.task = args.task
        self.setting = args.setting
        self.lang = args.lang
        self.model_name_or_path = args.model_name_or_path
        self.output_dir = args.output_dir
        train, eval, self.label_space = loadDataset(args.data_path, args.lang, args.setting)
        self.cat_to_term_dict, self.term_to_cat_dict, self.pol_to_term_dict, self.term_to_pol_dict, self.text_template, self.text_pattern, self.it_token = self.loadPhraseDicts('en')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        print("Device count: ", torch.cuda.device_count())
        self.gpu_count = torch.cuda.device_count()
        self.train, self.eval = self.preprocessData(train, self.tokenizer), self.preprocessData(eval, self.tokenizer)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)

    def loadPhraseDicts(self, lang):
        if lang in DATASETS:
        #     return CAT_TO_TERM_GERestaurant, TERM_TO_CAT_GERestaurant, POL_TO_TERM_GERestaurant, TERM_TO_POL_GERestaurant, TEXT_TEMPLATE_DE, TEXT_PATTERN_DE, IT_TOKEN_DE
        # elif dataset == 'rest-16':
            return globals()[f'CAT_TO_TERM_{lang.upper()}'], globals()[f'TERM_TO_CAT_{lang.upper()}'], globals()[f'POL_TO_TERM_{lang.upper()}'], globals()[f'TERM_TO_POL_{lang.upper()}'], globals()[f'TEXT_TEMPLATE_{lang.upper()}'], globals()[f'TEXT_PATTERN_{lang.upper()}'], globals()[f'IT_TOKEN_{lang.upper()}']
        else:
            print('Dataset language not valid.')
            
    def preprocessData(self, data, tokenizer):
        def labelToText(sample):
            if sample[2] != "NULL":
                aspect_term_text = sample[2]
            else:
                aspect_term_text = self.it_token
            ac_text = sample[0].capitalize()
            polarity_text = self.pol_to_term_dict[sample[1]]
            return self.text_template.format(ac_text=ac_text, polarity_text=polarity_text, aspect_term_text=aspect_term_text)

        def createOutput(samples):
            output_text = ''
            for sample in samples:
                output_text += labelToText(sample) + ' [SSEP] '
            return output_text[:-1]

        input_texts = data["text"].tolist()
        
        output_phrases = []

        for sample in list(data['labels']):
            output_phrases.append(createOutput(sample))
        
        input_encodings = tokenizer(input_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        output_encodings = tokenizer(output_phrases, padding=True, truncation=True, max_length=256, return_tensors="pt")
        
        return CustomDataset(input_encodings, output_encodings['input_ids'])

    def createModel(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)

    def computeMetrics(self, eval_pred):
        predictions, ground_truth = eval_pred

        predictions = np.where(predictions != -100,
                           predictions, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True)

        ground_truth = np.where(ground_truth != -100, ground_truth, self.tokenizer.pad_token_id)
        decoded_ground_truth = self.tokenizer.batch_decode(ground_truth, skip_special_tokens=True)

        print(decoded_preds[0])
        print(decoded_ground_truth[0])
        
        # print(decoded_preds)
        predictions_tuples = []
        for pred in decoded_preds:
            labels_extracted = []
            if pred != '':
                sentences = pred.split('[SSEP]')
                if len(sentences) > 0:
                    for label_strings in sentences:
                        match = re.match(self.text_pattern, label_strings.strip())

                        if match:
                            labels_extracted.append([match.group(1).lower(), self.term_to_pol_dict[match.group(2)] if match.group(2) in self.term_to_pol_dict else match.group(2), 'NULL' if match.group(3) == 'es' or match.group(3) == 'it' else match.group(3)])
            predictions_tuples.append(labels_extracted)

        ground_truth_tuples = []
        for gold in decoded_ground_truth:
            labels_extracted = []
            if gold != '':
                sentences = gold.split('[SSEP]')
                if len(sentences) > 0:
                    for label_strings in sentences:
                        match = re.match(self.text_pattern, label_strings.strip())

                        if match:
                            labels_extracted.append([match.group(1).lower(), self.term_to_pol_dict[match.group(2)] if match.group(2) in self.term_to_pol_dict else match.group(2), 'NULL' if match.group(3) == 'es' or match.group(3) == 'it' else match.group(3)])
            ground_truth_tuples.append(labels_extracted)

        print(predictions_tuples[0])
        print(ground_truth_tuples[0])
        
        self.predictions, self.false_predictions = convertLabels(predictions_tuples, self.task, self.label_space)
        ground_truths, _ = convertLabels(ground_truth_tuples, self.task, self.label_space)

        print(self.predictions[0])
        print(ground_truths[0])

        self.results = createResults(self.predictions, ground_truths, self.label_space, self.task)
        return self.results[4]

    def trainModel(self, lr, num_train_epochs, per_device_train_batch_size, args):

        training_args = Seq2SeqTrainingArguments(
            output_dir="outputs",
            learning_rate=lr,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=16,
            evaluation_strategy="no",
            save_strategy="no",
            logging_dir="logs",
            logging_steps=100,
            logging_strategy="epoch",
            bf16=True,
            report_to="none",
            predict_with_generate=True,
            generation_max_length=256,
            weight_decay=0.01,
            gradient_accumulation_steps = args.gradient_accumulation_steps
        )

        trainer = Seq2SeqTrainer(
            model_init=self.createModel,
            args=training_args,
            train_dataset=self.train,
            eval_dataset=self.eval,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.computeMetrics
        )
        
        print("Using the following hyperparameters: lr=" + str(lr) + " - epochs=" + str(num_train_epochs) + " - batch=" + str(per_device_train_batch_size*args.gradient_accumulation_steps*self.gpu_count))

        trainer.train()

        return trainer

    def evalModel(self, trainer, results_path, test = True):

        # Save results as tsv
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)

        _ = trainer.evaluate()
        
        results_asp, results_asp_pol, results_pairs, results_pol, results_phrases = self.results
        pd.DataFrame.from_dict(results_asp).transpose().to_csv(results_path + 'metrics_asp.tsv', sep = "\t")
        pd.DataFrame.from_dict(results_asp_pol).transpose().to_csv(results_path + 'metrics_asp_pol.tsv', sep = "\t")
        pd.DataFrame.from_dict(results_pairs).transpose().to_csv(results_path + 'metrics_pairs.tsv', sep = "\t")
        pd.DataFrame.from_dict(results_pol).transpose().to_csv(results_path + 'metrics_pol.tsv', sep = "\t")
        pd.DataFrame.from_dict(results_phrases).transpose().to_csv(results_path + 'metrics_phrases.tsv', sep = "\t")

        with open(results_path + 'config.txt', 'w') as f:
            for k,v in vars(trainer.args).items():
                f.write(f"{k}: {v}\n")
        
        # Save outputs to file
        with open(results_path + 'predictions.txt', 'w') as f:
            for line in self.predictions:
                f.write(f"{str(line).encode('utf-8')}\n")

        # Save false output labels to file
        if(len(self.false_predictions) > 0):
            with open(results_path + 'false_predictions.txt', 'w') as f:
                for line in self.false_predictions:
                    f.write(f"{str(line).encode('utf-8')}\n")
                
    def train_eval(self, args):
        results_path = f'{self.output_dir}{self.task}_{self.lang}_{self.setting}_{round(args.learning_rate,9)}_{args.per_device_train_batch_size}_{args.num_train_epochs}/' 
        
        trainer = self.trainModel(args.learning_rate, args.num_train_epochs, int(args.per_device_train_batch_size/(self.gpu_count * args.gradient_accumulation_steps)), args)

        self.evalModel(trainer, results_path)

if __name__ == "__main__":

    config = Config()
    # hfparser = transformers.HfArgumentParser([DataArgs, TrainingArgs])

    # data_config, training_config, extra_args = \
    #     hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    # args = argparse.Namespace(
    #     **vars(data_config),
    #     **vars(training_config)
    # )
        
    absa = ParaphraseABSA(config)
    absa.train_eval(config)


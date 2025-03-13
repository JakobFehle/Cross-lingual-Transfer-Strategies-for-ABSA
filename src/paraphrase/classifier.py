import pandas as pd
import numpy as np
import torch
import os, sys
import transformers
import argparse
import re
import json

utils = os.path.abspath('../src/utils/') # Relative path to utils scripts
sys.path.append(utils)

from config import Config
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorForSeq2Seq
from preprocessing import loadDataset
from evaluation import createResults, convertLabels, extractAspects
from preprocessing import loadDataset, CATEGORY_MAPPINGS, POLARITY_MAPPINGS_POL_TO_TERM, POLARITY_MAPPINGS_TERM_TO_POL, TEXT_TEMPLATES, TEXT_PATTERNS, IT_TOKENS, OUTPUT_KEYS

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class ABSADataset(TorchDataset):
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
        self.args = args
        self.task = args.task
        self.model_name_or_path = args.model_name_or_path
        self.t5model_class, self.t5tokenizer_class = self.getClassifiers(args.lang if (args.lang_setting == 'adapted' and args.data_setting == 'orig') else 'multi')

        print(f"Creating model using tokenizer: {self.t5tokenizer_class.__name__}")
        self.tokenizer = self.t5tokenizer_class.from_pretrained(self.model_name_or_path)
        self.gpu_count = torch.cuda.device_count()

        self.pol_to_term, self.term_to_pol, self.text_template, self.text_pattern, self.it_token, self.cat_from_en, self.cat_to_en = self.loadPhraseDicts(args.lang if args.lang_setting == 'adapted' else 'en')
        
        train, evaluation, self.label_space = self.splitForEvalSetting(loadDataset(args.data_path, args.lang, args.data_setting), args.eval_type)
        self.train, self.evaluation = self.preprocessData(train), self.preprocessData(evaluation)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
        
        print(f"Device count: {self.gpu_count}")

    def getClassifiers(self, lang):
        """Returns the correct tokenizer/model class for the selected language."""
        return {
            'en': [T5ForConditionalGeneration, T5Tokenizer],
            'de': [T5ForConditionalGeneration, T5Tokenizer],
            'fr': [T5ForConditionalGeneration, T5Tokenizer],
            'cs': [MT5ForConditionalGeneration, MT5Tokenizer],
            'tr': [MT5ForConditionalGeneration, MT5Tokenizer],
            'es': [T5ForConditionalGeneration, T5Tokenizer],
            'ru': [T5ForConditionalGeneration, T5Tokenizer],
            'nl': [T5ForConditionalGeneration, T5Tokenizer],
            'multi': [MT5ForConditionalGeneration, MT5Tokenizer]
        }[lang]

    def splitForEvalSetting(self, dataset, eval_type):
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
    
    def loadPhraseDicts(self, lang):
        """Loads language-specific dictionaries for category mapping and aspect terms."""

        ## Safety check
        for l in ["DE", "FR", "ES", "NL", "CS", "RU", "TR"]:
            en_to_x = CATEGORY_MAPPINGS[f'CAT_EN_{l.upper()}']
            x_to_en = CATEGORY_MAPPINGS[f'CAT_{l.upper()}_EN']
    
            for en_term, x_term in en_to_x.items():
                # Check if reverse mapping exists
                if x_term not in x_to_en or x_to_en[x_term] != en_term:
                    print(f"Mismatch in {lang}: '{en_term}' -> '{x_term}', but reverse is '{x_to_en.get(x_term, 'MISSING')}'")
        
        return POLARITY_MAPPINGS_POL_TO_TERM[lang], POLARITY_MAPPINGS_TERM_TO_POL[lang], TEXT_TEMPLATES[lang], TEXT_PATTERNS[lang], IT_TOKENS[lang], CATEGORY_MAPPINGS[f'CAT_EN_{lang.upper()}'], CATEGORY_MAPPINGS[f'CAT_{lang.upper()}_EN']

            
    def preprocessData(self, data):
        """Tokenizes text and processes labels into T5 training format."""
        def labelToText(sample):
            aspect_term_text = sample[2] if sample[2] != "NULL" else self.it_token
            return self.text_template.format(
                ac_text=self.cat_from_en.get(sample[0], sample[0]),
                polarity_text=self.pol_to_term.get(sample[1], sample[1]),
                aspect_term_text=aspect_term_text
            )

        input_texts = data["text"].tolist()
        output_texts = [' [SSEP] '.join(map(labelToText, labels)) for labels in data['labels']]

        print('Train dataset snippet:\n')
        print(input_texts[10])
        print(output_texts[10])
        
        input_encodings = self.tokenizer(input_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        output_encodings = self.tokenizer(output_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        
        return ABSADataset(input_encodings, output_encodings['input_ids'])

    def createModel(self):
        """Initializes and loads the model."""
        print('Creating model of type ', self.t5model_class)
        print('Loading model ', self.model_name_or_path)
        return self.t5model_class.from_pretrained(self.model_name_or_path)

    def extractLabels(self, paraphrased_text):
        extracted_labels = []
        sents = [s.strip() for s in paraphrased_text.split('[SSEP]')]
        for sent in sents:
                match = re.match(self.text_pattern, sent)
                if match:
                    try:
                        extracted_labels.append([self.cat_to_en[match.group(1).strip()], self.term_to_pol[match.group(2)], 'NULL' if match.group(3) == self.it_token else match.group(3)])
                    except KeyError:
                        print(f"Extraction error for sentence: '{sent}'")
                        print(f"Matched phrases: '{match}'")
        return extracted_labels

    def computeMetrics(self, eval_pred):
        predictions, ground_truth = eval_pred

        predictions_raw = np.where(predictions != -100,
                           predictions, self.tokenizer.pad_token_id)

        predictions_decoded = self.tokenizer.batch_decode(
            predictions_raw, skip_special_tokens=True)

        ground_truth = np.where(ground_truth != -100, ground_truth, self.tokenizer.pad_token_id)
        ground_truth_decoded = self.tokenizer.batch_decode(ground_truth, skip_special_tokens=True)

        print('Test dataset prediction:\n')
        print('Gold: ', ground_truth_decoded[10])
        print('Pred: ', predictions_decoded[10])
        
        # print(decoded_preds)
        predictions = []
        for prediction in predictions_decoded:
            predictions.append(self.extractLabels(prediction))

        self.predictions = predictions
        
        gold = []
        for ground_truth in ground_truth_decoded:
            gold.append(self.extractLabels(ground_truth))
        
        predictions, self.false_predictions = convertLabels(predictions, self.task, self.label_space)
        gold, _ = convertLabels(gold, self.task, self.label_space)

        self.results = createResults(predictions, gold, self.label_space, self.task)
        return self.results[4]

    def trainModel(self):

        adjusted_batch = int(self.args.per_device_train_batch_size/(self.gpu_count * self.args.gradient_accumulation_steps))
        
        training_args = Seq2SeqTrainingArguments(
            output_dir = 'paraphrase/outputs/',
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=adjusted_batch,
            per_device_eval_batch_size=16,
            evaluation_strategy="no",
            save_strategy="no",
            logging_dir="logs",
            logging_steps=100,
            logging_strategy="epoch",
            report_to="none",
            predict_with_generate=True,
            generation_max_length=256,
            weight_decay=0.01,
            gradient_accumulation_steps = self.args.gradient_accumulation_steps
        )

        trainer = Seq2SeqTrainer(
            model_init=self.createModel,
            args=training_args,
            train_dataset=self.train,
            eval_dataset=self.evaluation,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.computeMetrics
        )
        
        print("Using the following hyperparameters: lr=" + str(self.args.learning_rate) + " - epochs=" + str(self.args.num_train_epochs) + " - batch=" + str(self.args.per_device_train_batch_size*self.gpu_count))

        trainer.train()

        return trainer

    def evaluate(self, trainer, results_path):
        
        # Save results as tsv
        os.makedirs(results_path, exist_ok=True)

        results = trainer.evaluate()
        
        for idx, name in enumerate(["asp", "asp_pol", "pairs", "pol", "phrases"]):
            pd.DataFrame.from_dict(self.results[idx]).transpose().to_csv(f"{results_path}metrics_{name}.tsv", sep="\t")

        with open(os.path.join(results_path, 'config.json'), "w", encoding="utf-8") as f:
            trainer_args = {key: vars(trainer.args)[key] for key in OUTPUT_KEYS}

            trainer_args.update({
                "model_name": self.model_name_or_path,
                "task": self.task,
                "data_setting": self.args.data_setting,
                "lang": self.args.lang,
                "lang_setting": self.args.lang_setting,
                "eval_type": self.args.eval_type
            })
            json.dump(trainer_args, f, indent=4, ensure_ascii=False)
        
        with open(os.path.join(results_path, 'predictions.json'), "w", encoding="utf-8") as f:
            json.dump({'predictions': self.predictions}, f, indent=4, ensure_ascii=False)

        # Save false output labels to file
        if(len(self.false_predictions) > 0):
            with open(results_path + 'false_predictions.txt', 'w') as f:
                for line in self.false_predictions:
                    f.write(f"{str(line).encode('utf-8')}\n")
                
    def train_eval(self):
        results_path = f"{self.args.output_dir}{self.task}_{self.args.lang}_{self.args.lang_setting}_{self.args.eval_type.replace('_', '-')}_{self.args.data_setting}_{round(self.args.learning_rate,9)}_{self.args.per_device_train_batch_size}_{self.args.num_train_epochs}_{self.args.seed}/"

        trainer = self.trainModel()

        self.evaluate(trainer, results_path)
if __name__ == "__main__":

    config = Config()
    set_seed(config.seed)
        
    absa = ParaphraseABSA(config)
    absa.train_eval()


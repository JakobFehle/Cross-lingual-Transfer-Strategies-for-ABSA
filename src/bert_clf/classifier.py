import pandas as pd
import numpy as np
import torch
import os, sys
import json
import time

utils = os.path.abspath('../src/utils/') # Relative path to utils scripts
sys.path.append(utils)

from config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from scipy.special import expit
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorWithPadding
from preprocessing import loadDataset, splitForEvalSetting
from evaluation import createResults, convertLabels
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class ABSADataset(TorchDataset):
    """Dataset class for Aspect-Based Sentiment Analysis (ABSA)."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["label"] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

class MultiLabelABSA:
    """Multi-label ABSA model training and evaluation."""
    
    def __init__(self, args):
        self.args = args
        self.task = args.task
        self.model_name_or_path = args.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.gpu_count = torch.cuda.device_count()
        
        self.model = self.createModel(len(self.train[0]['label']))
        
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        print("Device count: ", torch.cuda.device_count())
    
    def preprocessData(self, data, train = False):
        """Tokenizes text and processes labels."""
        texts = data["text"].tolist()
        
        if self.task == 'acd':
            self.label_space = set([label.split(':')[0] for label in self.label_space])
        labels = [[label[0] if self.task == 'acd' else ':'.join(label[:2]) for label in labels] for labels in data['labels']]

        if train:
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit([self.label_space])
                    
        labels = torch.tensor(self.mlb.transform(labels), dtype=torch.float32)
        encodings = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        
        return ABSADataset(encodings, labels)

    def createModel(self, num_labels):
        """Initializes the model."""
        print('Creating Model: ', self.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        return model

    def computeMetrics(self, eval_pred):
        """Computes evaluation metrics for the model."""
        predictions, ground_truth = eval_pred
        
        predictions_raw = (expit(predictions) > 0.5)
        
        predictions_decoded = self.mlb.inverse_transform(predictions_raw)
        ground_truth_decoded = self.mlb.inverse_transform(ground_truth)
            
        pred_labels = [[p.split(':') if ':' in p else p for p in pred] for pred in predictions_decoded]
        gold_labels = [[g.split(':') if ':' in g else g for g in gt] for gt in ground_truth_decoded]
        
        pred_labels, _ = convertLabels(pred_labels, self.task, self.label_space)
        gold_labels, _ = convertLabels(gold_labels, self.task, self.label_space)
        
        results = createResults(pred_labels, gold_labels, self.label_space, self.task)

        return results, pred_labels, gold_labels

    def trainModel(self, train_dataset):
        """Trains the model with given hyperparameters."""

        adjusted_batch = int(self.args.per_device_train_batch_size/self.gpu_count)
        
        training_args = TrainingArguments(
            output_dir = 'bert_clf/outputs/',
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=adjusted_batch,
            per_device_eval_batch_size=16,
            evaluation_strategy="no",
            save_strategy="no",
            logging_dir="logs",
            logging_steps=100,
            logging_strategy="epoch",
            bf16=True,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.computeMetrics,
        )
        
        print("Using the following hyperparameters: lr=" + str(self.args.learning_rate) + " - epochs=" + str(self.args.num_train_epochs) + " - batch=" + str(self.args.per_device_train_batch_size))
        
        start_time = time.time()

        trainer.train()

        end_time = time.time()
        training_duration = end_time - start_time

        trainer_args = {}
        trainer_args.update({
            "model_name": self.model_name_or_path,
            "task": self.task,
            "data_setting": self.args.data_setting,
            "lang": self.args.lang,
            "lang_setting": self.args.lang_setting,
            "per_device_train_batch_size": self.args.per_device_train_batch_size,
            "learning_rate": self.args.learning_rate,
            "num_train_epochs": self.args.num_train_epochs,
            "eval_type": self.args.eval_type,
            "train_runtime": training_duration
        })

        return trainer, trainer_args


    def savePredictions(self, result, preds, golds, args, config, eval_type):
        """Evaluates the model and saves results."""

        output_path = f"{args.output_dir}{args.task}_{args.lang}_{args.lang_setting}_{args.eval_type}-{eval_type[0]}_{args.data_setting}_{round(args.learning_rate,9)}_{args.per_device_train_batch_size}_{args.num_train_epochs}_{args.seed}/" 
        os.makedirs(output_path, exist_ok=True)
        
        if self.task == 'acd':
            pd.DataFrame.from_dict(result[0]).transpose().to_csv(f"{output_path}metrics_asp.tsv", sep="\t")
                
        else:
            for idx, name in enumerate(["asp", "asp_pol", "pairs", "pol"]):
                pd.DataFrame.from_dict(result[idx]).transpose().to_csv(f"{output_path}metrics_{name}.tsv", sep="\t")

        try:
            matched_samples = [
                {"predictions": pred, "gold_labels": gold}
                for pred, gold in zip(preds, golds)
            ]
            print(matched_samples[:5])
            with open(os.path.join(output_path, 'predictions.json'), "w", encoding="utf-8") as f:
                json.dump({"test": matched_samples}, f, indent=4, ensure_ascii=False)

        except:
            pass

        with open(os.path.join(output_path, 'config.json'), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
                
    def train_eval(self):

        args = self.args

        train_df, eval_df, self.label_space = self.splitForEvalSetting(loadDataset(args.data_path, args.lang, args.data_setting), args.eval_type)
        train = self.preprocessData(train_df, True)

        if args.data_setting == 'balanced':
            
            eval_balanced = self.preprocessData(eval_df)

            trainer, trainer_args = self.trainModel(train)

            if 'eval' in args.eval_type:

                results_balanced, preds, golds = trainer.evaluate(eval_dataset=eval_balanced)
                self.savePredictions(results_balanced, preds, golds, args, trainer_args, 'balanced')
            
            else:

                results_balanced, preds, golds = trainer.evaluate(eval_dataset=eval_balanced)
                self.savePredictions(results_balanced, preds, golds, args, trainer_args, 'balanced')

                _, eval_df, _ = self.splitForEvalSetting(loadDataset(args.data_path, args.lang, 'orig'), args.eval_type)
                eval_orig = self.preprocessData(eval_df)
                results_orig, preds, golds = trainer.evaluate(eval_dataset=eval_orig)
                self.savePredictions(results_orig, preds, golds, args, trainer_args, 'orig')
        
        else:

            eval_orig = self.preprocessData(eval_df)

            trainer, trainer_args = self.trainModel(train)

            if 'eval' in args.eval_type:

                results_orig, preds, golds = trainer.evaluate(eval_dataset=eval_orig)
                self.savePredictions(results_orig, preds, golds, args, trainer_args, 'orig')

            else:
                if args.lang != 'tr':
                    _, eval_df, _ = self.splitForEvalSetting(loadDataset(args.data_path, args.lang, 'balanced'), args.eval_type)
                    eval_balanced = self.preprocessData(eval_df)

                    results_balanced, preds, golds = trainer.evaluate(eval_dataset=eval_balanced)
                    self.savePredictions(results_balanced, preds, golds, args, trainer_args, 'balanced')

                results_orig, preds, golds = trainer.evaluate(eval_dataset=eval_orig)
                self.savePredictions(results_orig, preds, golds, args, trainer_args, 'orig')
        

if __name__ == "__main__":
    config = Config()
    set_seed(config.seed)
    
    absa = MultiLabelABSA(config)  
    absa.train_eval()


import pandas as pd
import numpy as np
import torch
import os, sys
import json

utils = os.path.abspath('../src/utils/') # Relative path to utils scripts
sys.path.append(utils)

from config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from scipy.special import expit
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorWithPadding
from preprocessing import loadDataset
from evaluation import createResults, convertLabels
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")

OUTPUT_KEYS = ['per_device_train_batch_size', 'gradient_accumulation_steps', 'learning_rate', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'max_grad_norm', 'num_train_epochs', 'lr_scheduler_type', 'warmup_steps', 'seed', 'bf16', 'fp16', 'group_by_length', '_n_gpu']

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
        
        train, evaluation, self.label_space = self.splitForEvalSetting(loadDataset(args.data_path, args.lang, args.data_setting), args.eval_type)
        self.train, self.evaluation = self.preprocessData(train, True), self.preprocessData(evaluation)
        
        self.model = self.createModel(len(self.train[0]['label']))
        
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        print("Device count: ", torch.cuda.device_count())

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

        self.predictions = pred_labels
        
        pred_labels, _ = convertLabels(pred_labels, self.task, self.label_space)
        gold_labels, _ = convertLabels(gold_labels, self.task, self.label_space)
        
        self.results = createResults(pred_labels, gold_labels, self.label_space, self.task)

        return self.results[0] if self.task == 'acd' else self.results[1]

    def trainModel(self):
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
            train_dataset=self.train,
            eval_dataset=self.evaluation,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.computeMetrics,
        )
        
        print("Using the following hyperparameters: lr=" + str(self.args.learning_rate) + " - epochs=" + str(self.args.num_train_epochs) + " - batch=" + str(self.args.per_device_train_batch_size))
        
        trainer.train()

        return trainer


    def savePredictions(self, trainer, results_path, test = True):
        """Evaluates the model and saves results."""
        # Save results as tsv
        os.makedirs(results_path, exist_ok=True)
        results = trainer.evaluate()
        
        if self.task == 'acd':
            pd.DataFrame.from_dict(self.results[0]).transpose().to_csv(f"{results_path}metrics_asp.tsv", sep="\t")
                
        else:
            for idx, name in enumerate(["asp", "asp_pol", "pairs", "pol"]):
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
                
    def train_eval(self):

        trainer = self.trainModel()

        results_path = f"{self.args.output_dir}{self.task}_{self.args.lang}_{self.args.lang_setting}_{self.args.eval_type.replace('_', '-')}_{self.args.data_setting}_{round(self.args.learning_rate,9)}_{self.args.per_device_train_batch_size}_{self.args.num_train_epochs}_{self.args.seed}/" 
        
        self.savePredictions(trainer, results_path)

if __name__ == "__main__":
    config = Config()
    set_seed(config.seed)
    
    absa = MultiLabelABSA(config)  
    absa.train_eval()


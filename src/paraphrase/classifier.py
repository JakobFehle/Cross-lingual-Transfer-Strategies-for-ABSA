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
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")
warnings.filterwarnings("ignore", category=UserWarning)

DATASETS = ['nl', 'de', 'en', 'cs', 'ru', 'fr', 'es']

CATEGORY_MAPPINGS = {
"CAT_EN_EN": {
    "restaurant general": "restaurant general",
    "restaurant miscellaneous": "restaurant miscellaneous",
    "restaurant prices": "restaurant prices",
    "food quality": "food quality",
    "food prices": "food prices",
    "food style_options": "food style_options",
    "drinks quality": "drinks quality",
    "drinks prices": "drinks prices",
    "drinks style_options": "drinks style_options",
    "location general": "location general",
    "service general": "service general",
    "ambience general": "ambience general"
  },
"CAT_EN_RU": {
    "restaurant general": "Ресторан Общее",
    "restaurant miscellaneous": "Ресторан Разное",
    "restaurant prices": "Ресторан Цены",
    "food quality": "Еда Качество",
    "food prices": "Еда Цены",
    "food style_options": "Еда Стиль",
    "drinks quality": "Напитки Качество",
    "drinks prices": "Напитки Цены",
    "drinks style_options": "Напитки Стиль",
    "location general": "Расположение Общее",
    "service general": "Обслуживание Общее",
    "ambience general": "Атмосфера Общее"
  },
"CAT_RU_EN": {
    "Ресторан Общее": "restaurant general",
    "Ресторан Разное": "restaurant miscellaneous",
    "Ресторан Цены": "restaurant prices",
    "Еда Качество": "food quality",
    "Еда Цены": "food prices",
    "Еда Стиль": "food style_options",
    "Напитки Качество": "drinks quality",
    "Напитки Цены": "drinks prices",
    "Напитки Стиль": "drinks style_options",
    "Расположение Общее": "location general",
    "Обслуживание Общее": "service general",
    "Атмосфера Общее": "ambience general"
  },
"CAT_EN_CS": {
    "restaurant general": "Restaurace Obecné",
    "restaurant miscellaneous": "Restaurace Různé",
    "restaurant prices": "Restaurace Ceny",
    "food quality": "Jídlo Kvalita",
    "food prices": "Jídlo Ceny",
    "food style_options": "Jídlo Styl",
    "drinks quality": "Nápoje Kvalita",
    "drinks prices": "Nápoje Ceny",
    "drinks style_options": "Nápoje Styl",
    "location general": "Poloha Obecná",
    "service general": "Služby Obecné",
    "ambience general": "Prostředí Obecné"
  },
"CAT_CS_EN": {
    "Restaurace Obecné": "restaurant general",
    "Restaurace Různé": "restaurant miscellaneous",
    "Restaurace Ceny": "restaurant prices",
    "Jídlo Kvalita": "food quality",
    "Jídlo Ceny": "food prices",
    "Jídlo Styl": "food style_options",
    "Nápoje Kvalita": "drinks quality",
    "Nápoje Ceny": "drinks prices",
    "Nápoje Styl": "drinks style_options",
    "Poloha Obecná": "location general",
    "Služby Obecné": "service general",
    "Prostředí Obecné": "ambience general"
  },
"CAT_EN_NL": {
    "restaurant general": "Restaurant Algemeen",
    "restaurant miscellaneous": "Restaurant Diversen",
    "restaurant prices": "Restaurant Prijzen",
    "food quality": "Voedsel Kwaliteit",
    "food prices": "Voedsel Prijzen",
    "food style_options": "Voedsel Stijlopties",
    "drinks quality": "Dranken Kwaliteit",
    "drinks prices": "Dranken Prijzen",
    "drinks style_options": "Dranken Stijlopties",
    "location general": "Locatie Algemeen",
    "service general": "Service Algemeen",
    "ambience general": "Sfeer Algemeen"
  },
"CAT_NL_EN": {
    "Restaurant Algemeen": "restaurant general",
    "Restaurant Diversen": "restaurant miscellaneous",
    "Restaurant Prijzen": "restaurant prices",
    "Voedsel Kwaliteit": "food quality",
    "Voedsel Prijzen": "food prices",
    "Voedsel Stijlopties": "food style_options",
    "Dranken Kwaliteit": "drinks quality",
    "Dranken Prijzen": "drinks prices",
    "Dranken Stijlopties": "drinks style_options",
    "Locatie Algemeen": "location general",
    "Service Algemeen": "service general",
    "Sfeer Algemeen": "ambience general"
  },
"CAT_EN_ES": {
    "restaurant general": "Restaurante General",
    "restaurant miscellaneous": "Restaurante Varios",
    "restaurant prices": "Restaurante Precios",
    "food quality": "Comida Calidad",
    "food prices": "Comida Precios",
    "food style_options": "Comida Estilo",
    "drinks quality": "Bebidas Calidad",
    "drinks prices": "Bebidas Precios",
    "drinks style_options": "Bebidas Estilo",
    "location general": "Ubicación General",
    "service general": "Servicio General",
    "ambience general": "Ambiente General"
  },
"CAT_ES_EN": {
    "Restaurante General": "restaurant general",
    "Restaurante Varios": "restaurant miscellaneous",
    "Restaurante Precios": "restaurant prices",
    "Comida Calidad": "food quality",
    "Comida Precios": "food prices",
    "Comida Estilo": "food style_options",
    "Bebidas Calidad": "drinks quality",
    "Bebidas Precios": "drinks prices",
    "Bebidas Estilo": "drinks style_options",
    "Ubicación General": "location general",
    "Servicio General": "service general",
    "Ambiente General": "ambience general"
  },
"CAT_EN_FR": {
    "restaurant general": "Restaurant Général",
    "restaurant miscellaneous": "Restaurant Divers",
    "restaurant prices": "Restaurant Prix",
    "food quality": "Nourriture Qualité",
    "food prices": "Nourriture Prix",
    "food style_options": "Nourriture Style",
    "drinks quality": "Boissons Qualité",
    "drinks prices": "Boissons Prix",
    "drinks style_options": "Boissons Style",
    "location general": "Emplacement Général",
    "service general": "Service Général",
    "ambience general": "Ambiance Général"
  },
"CAT_FR_EN": {
    "Restaurant Général": "restaurant general",
    "Restaurant Divers": "restaurant miscellaneous",
    "Restaurant Prix": "restaurant prices",
    "Nourriture Qualité": "food quality",
    "Nourriture Prix": "food prices",
    "Nourriture Style": "food style_options",
    "Boissons Qualité": "drinks quality",
    "Boissons Prix": "drinks prices",
    "Boissons Style": "drinks style_options",
    "Emplacement Général": "location general",
    "Service Général": "service general",
    "Ambiance Général": "ambience general"
  },
"CAT_EN_DE": {
    "restaurant general": "Restaurant Allgemein",
    "restaurant miscellaneous": "Restaurant Sonstiges",
    "restaurant prices": "Restaurant Preise",
    "food quality": "Essen Qualität",
    "food prices": "Essen Preise",
    "food style_options": "Essen Stiloptionen",
    "drinks quality": "Getränke Qualität",
    "drinks prices": "Getränke Preise",
    "drinks style_options": "Getränke Stiloptionen",
    "location general": "Lage Allgemein",
    "service general": "Service Allgemein",
    "ambience general": "Ambiente Allgemein"
  },
"CAT_DE_EN": {
    "Restaurant Allgemein": "restaurant general",
    "Restaurant Sonstiges": "restaurant miscellaneous",
    "Restaurant Preise": "restaurant prices",
    "Essen Qualität": "food quality",
    "Essen Preise": "food prices",
    "Essen Stiloptionen": "food style_options",
    "Getränke Qualität": "drinks quality",
    "Getränke Preise": "drinks prices",
    "Getränke Stiloptionen": "drinks style_options",
    "Lage Allgemein": "location general",
    "Service Allgemein": "service general",
    "Ambiente Allgemein": "ambience general"
  },
"CAT_EN_TR": {
    "restaurant general": "Restoran Genel",
    "restaurant miscellaneous": "Restoran Çeşitli",
    "restaurant prices": "Restoran Fiyatlar",
    "food quality": "Yemek Kalite",
    "food prices": "Yemek Fiyatlar",
    "food style_options": "Yemek Tarz",
    "drinks quality": "İçecek Kalite",
    "drinks prices": "İçecek Fiyatlar",
    "drinks style_options": "İçecek Tarz",
    "location general": "Konum Genel",
    "service general": "Hizmet Genel",
    "ambience general": "Ortam Genel"
  },
"CAT_TR_EN": {
    "Restoran Genel": "restaurant general",
    "Restoran Çeşitli": "restaurant miscellaneous",
    "Restoran Fiyatlar": "restaurant prices",
    "Yemek Kalite": "food quality",
    "Yemek Fiyatlar": "food prices",
    "Yemek Tarz": "food style_options",
    "İçecek Kalite": "drinks quality",
    "İçecek Fiyatlar": "drinks prices",
    "İçecek Tarz": "drinks style_options",
    "Konum Genel": "location general",
    "Hizmet Genel": "service general",
    "Ortam Genel": "ambience general"
  }
}
POLARITY_MAPPINGS_POL_TO_TERM = {
    "en": {"negative": "bad", "neutral": "okay", "positive": "great"},
    "de": {"negative": "schlecht", "neutral": "okay", "positive": "gut"},
    "ru": {"negative": "плохо", "neutral": "нормально", "positive": "отлично"},
    "cs": {"negative": "špatné", "neutral": "ok", "positive": "skvělé"},
    "nl": {"negative": "slecht", "neutral": "oké", "positive": "geweldig"},
    "es": {"negative": "malo", "neutral": "ok", "positive": "genial"},
    "fr": {"negative": "mauvais", "neutral": "ok", "positive": "super"},
    "tr": {"negative": "kötü", "neutral": "tamam", "positive": "harika"}
}

POLARITY_MAPPINGS_TERM_TO_POL = {
    "en": {"bad": "negative", "okay": "neutral", "great": "positive"},
    "de": {"schlecht": "negative", "okay": "neutral", "gut": "positive"},
    "ru": {"плохо": "negative", "нормально": "neutral", "отлично": "positive"},
    "cs": {"špatné": "negative", "ok": "neutral", "skvělé": "positive"},
    "nl": {"slecht": "negative", "oké": "neutral", "geweldig": "positive"},
    "es": {"malo": "negative", "ok": "neutral", "genial": "positive"},
    "fr": {"mauvais": "negative", "ok": "neutral", "super": "positive"},
    "tr": {"kötü": "negative", "tamam": "neutral", "harika": "positive"}
}

OUTPUT_KEYS = ['per_device_train_batch_size', 'gradient_accumulation_steps', 'learning_rate', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'max_grad_norm', 'num_train_epochs', 'lr_scheduler_type', 'warmup_steps', 'seed', 'bf16', 'fp16', 'group_by_length', '_n_gpu', 'generation_max_length']

TEXT_TEMPLATES = {
    "en": "{ac_text} is {polarity_text} because {aspect_term_text} is {polarity_text}",
    "de": "{ac_text} ist {polarity_text} weil {aspect_term_text} {polarity_text} ist",
    "ru": "{ac_text} {polarity_text} потому что {aspect_term_text} {polarity_text}",
    "cs": "{ac_text} je {polarity_text} protože {aspect_term_text} je {polarity_text}",
    "nl": "{ac_text} is {polarity_text} omdat {aspect_term_text} {polarity_text} is",
    "es": "{ac_text} es {polarity_text} porque {aspect_term_text} es {polarity_text}",
    "fr": "{ac_text} est {polarity_text} parce que {aspect_term_text} est {polarity_text}",
    "tr": "{ac_text} {polarity_text} çünkü {aspect_term_text} {polarity_text}"
}

TEXT_PATTERNS = {
    "en": r"(.*) is (.*) because (.*) is (.*)",
    "de": r"(.*) ist (.*) weil (.*) (.*) ist",
    "ru": r"(.*) (.*) потому что (.*) (.*)",
    "cs": r"(.*) je (.*) protože (.*) je (.*)",
    "nl": r"(.*) is (.*) omdat (.*) (.*) is",
    "es": r"(.*) es (.*) porque (.*) es (.*)",
    "fr": r"(.*) est (.*) parce que (.*) est (.*)",
    "tr": r"(.*) (.*) çünkü (.*) (.*)"
}
 
IT_TOKENS = {"en": "it", "de": "es", "ru": "это", "cs": "to", "nl": "het", "es": "eso", "fr": "ça", "tr": "bu"}

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
        for l in ["DE", "FR", "ES", "NL", "CS", "RU", "TR"]:
            en_to_x = CATEGORY_MAPPINGS[f'CAT_EN_{l.upper()}']
            x_to_en = CATEGORY_MAPPINGS[f'CAT_{l.upper()}_EN']
    
            for en_term, x_term in en_to_x.items():
                # Check if reverse mapping exists
                if x_term not in x_to_en or x_to_en[x_term] != en_term:
                    print(f"❌ Mismatch in {lang}: '{en_term}' -> '{x_term}', but reverse is '{x_to_en.get(x_term, 'MISSING')}'")
        
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

    def savePredictions(self, trainer, results_path):
        
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

        self.savePredictions(trainer, results_path)
if __name__ == "__main__":

    config = Config()
    set_seed(config.seed)
        
    absa = ParaphraseABSA(config)
    absa.train_eval()


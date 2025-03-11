import pandas as pd
import re
import json
import torch
import os

from itertools import product
from ast import literal_eval

LABEL_SPACES = ['ambience general:positive', 'ambience general:neutral', 'ambience general:negative', 'drinks prices:positive', 'drinks prices:neutral', 'drinks prices:negative', 'drinks quality:positive', 'drinks quality:neutral', 'drinks quality:negative', 'drinks style_options:positive', 'drinks style_options:neutral', 'drinks style_options:negative', 'food prices:positive', 'food prices:neutral', 'food prices:negative', 'food quality:positive', 'food quality:neutral', 'food quality:negative', 'food style_options:positive', 'food style_options:neutral', 'food style_options:negative', 'location general:positive', 'location general:neutral', 'location general:negative', 'restaurant general:positive', 'restaurant general:neutral', 'restaurant general:negative', 'restaurant miscellaneous:positive', 'restaurant miscellaneous:neutral', 'restaurant miscellaneous:negative', 'restaurant prices:positive', 'restaurant prices:neutral', 'restaurant prices:negative', 'service general:positive', 'service general:neutral', 'service general:negative']

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

REGEX_ASPECTS_ACSD = r"\(([^,]+),[^,]+,\s*\"[^\"]*\"\)"
REGEX_LABELS_ACSD = r"\([^,]+,\s*([^,]+)\s*,\s*\"[^\"]*\"\s*\)"
REGEX_PHRASES_ACSD = r"\([^,]+,\s*[^,]+\s*,\s*\"([^\"]*)\"\s*\)"
REGEX_ASPECTS = r"\(([^,]+),[^)]+\)"
REGEX_LABELS = r"\([^,]+,\s*([^)]+)\)"

LANGS = ['nl', 'de', 'en', 'cs', 'ru', 'fr', 'es']

def raise_err(ex):
    raise ex

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

def loadDataset(data_path, language, setting):

    setting_str = '_b' if setting == 'balanced' else ''
    
    label_space = LABEL_SPACES

    if 'multi' not in setting:
        path_train = f'{data_path}/{language}/train{setting_str}.json'
        path_eval = f'{data_path}/{language}/test{setting_str}.json'
        
        df_train = pd.read_json(path_train, orient="records", lines=True).set_index('id')
        df_eval = pd.read_json(path_eval, orient="records", lines=True).set_index('id')

    else:
        dfs = []
        for lang in [l for l in LANGS if l != language]:
            path_train = f'{data_path}/{lang}/train_b.json'
            dfs.append(pd.read_json(path_train, orient="records", lines=True).set_index('id'))

        if setting == 'multi_id':
            path_train = f'{data_path}/{language}/train_b.json'
            dfs.append(pd.read_json(path_train, orient="records", lines=True).set_index('id'))
            
        df_train = pd.concat(dfs)
        
        path_eval = f'{data_path}/{language}/test{setting_str}.json'
        df_eval = pd.read_json(path_eval, orient="records", lines=True).set_index('id')

    print(f'Loading dataset ...')
    print(f'Dataset language: {language}')
    print(f'Setting: {setting}')
    print(f'Train Length: ', len(df_train))
    print(f'Eval Length: ', len(df_eval))
        
    return df_train, df_eval, label_space

def createPromptText(lang, prompt_templates, prompt_style, example_text, example_labels, dataset_name = 'rest-16', absa_task = 'acd', train = False):

    # Set templates based on prompt config
    if dataset_name not in ['GERestaurant', 'rest-16']:
        raise NotImplementedError('Prompt template not found: Dataset name not valid.')
    else:
        dataset_name = dataset_name.replace('-','')[:6]

    if lang not in ['en', 'ger']:
        raise NotImplementedError('Prompt template not found: Prompt language not valid.')

    if absa_task not in ['acd', 'acsa', 'e2e', 'e2e-e', 'tasd']:
        raise NotImplementedError('Prompt template not found: Absa task not valid.')
    
    if prompt_style not in ['basic', 'context', 'cot']:
        raise NotImplementedError('Prompt template not found: Prompt style not valid.')
    else:
        template_prompt_style = prompt_style if prompt_style != 'cot' else 'context'

    try:
        prompt_template = prompt_templates[f'PROMPT_TEMPLATE_{lang.upper()}_{"E2E" if absa_task == "e2e-e" else absa_task.upper()}_{dataset_name.upper()}_{template_prompt_style.upper()}']
    except:
        raise NotImplementedError('Prompt template does not exist.')
    
    

    if example_labels is not None:
        try:
            reg_asp = re.compile(REGEX_ASPECTS_ACSD)
            reg_lab = re.compile(REGEX_LABELS_ACSD)
            reg_phr = re.compile(REGEX_PHRASES_ACSD)
            
            # Extract Aspects from  sample
            re_aspects = [reg_asp.match(pair)[1] for pair in example_labels]
            
            # Extract Polarities from  sample
            re_labels = [reg_lab.match(pair)[1] for pair in example_labels]
        
            # Extract Aspect-Phrases from  sample
            re_phrases = [reg_phr.match(pair)[1] for pair in example_labels]
            
        except:
            try:
                reg_asp = re.compile(REGEX_ASPECTS)
                reg_lab = re.compile(REGEX_LABELS)
                
                # Extract Aspects from  sample
                re_aspects = [reg_asp.match(pair)[1] for pair in example_labels]
                
                # Extract Polarities from  sample
                re_labels = [reg_lab.match(pair)[1] for pair in example_labels]
        
            except:
                raise NotImplementedError("Data-Format is not ['(Aspect Category, Sentiment Polarity, Aspect Phrase)', '(Aspect Category, Sentiment Polarity, Aspect Phrase)']")
        
        if absa_task == 'acd':        
            example_labels = re_aspects
    
        elif absa_task == 'acsa':
            example_labels = [re_aspects, re_labels]
            
        elif absa_task == 'e2e' or absa_task == 'e2e-e':
            example_labels = [re_phrases, re_labels]
            
        elif absa_task == 'tasd':
            example_labels = [re_aspects, re_labels, re_phrases]

    prompt = '### Instruction:\n' + prompt_template[0] + ' ' + prompt_template[1] + '\n\n' + f'### Input:\n{example_text} \n\n### Output:\n' 

    # Determine if train or test prompt and append target label if necessary

    if absa_task == 'acd':
        target_text = '[' + ', '.join(example_labels) + ']'
    elif absa_task == 'acsa':
         target_text = '[' + ', '.join([f'({example_labels[0][i]}, {example_labels[1][i]})' for i in range(len(example_labels[0]))]) + ']'
    elif absa_task == 'e2e' or absa_task == 'e2e-e':
         target_text = '[' + ', '.join([f'("{example_labels[0][i]}", {example_labels[1][i]})' for i in range(len(example_labels[0]))]) + ']'
    elif absa_task == 'tasd':
         target_text = '[' + ', '.join([f'({example_labels[0][i]}, {example_labels[1][i]}, "{example_labels[2][i]}")' for i in range(len(example_labels[0]))]) + ']'

    if train:
        return prompt + target_text, None
    else:
        return prompt, target_text
        
def createPrompts(df_train, df_test, args, eos_token = ''):

    prompts_train = []
    prompts_test = []
    ground_truth_labels = []
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.dataset in ['rest-16', 'GERestaurant']:
        with open(os.path.join(base_dir, f'prompts_{args.dataset.replace("-","")}.json'), encoding='utf-8') as json_prompts:
             prompt_templates = json.load(json_prompts)
    else:
        raise NotImplementedError('Prompt-Template not found: File does not exist.')

    label_column = 'labels_phrases' if 'labels_phrases' in df_train.columns else 'labels' if 'labels' in df_train.columns else lambda x: raise_err(NotImplementedError('Dataset does not have column with label targets.'))

    for index, row in df_train.iterrows():       

        prompt, _ = createPromptText(lang = args.lang, prompt_templates = prompt_templates, prompt_style = args.prompt_style, example_text = row['text'], example_labels = row[label_column], dataset_name = args.dataset, absa_task = args.task, train = True) 

        prompts_train.append(prompt + eos_token)

    for index, row in df_test.iterrows():
        prompt, targets = createPromptText(lang = args.lang, prompt_templates = prompt_templates, prompt_style = args.prompt_style, example_text = row['text'], example_labels = row[label_column], dataset_name = args.dataset, absa_task = args.task)
        
        prompts_test.append(prompt + eos_token)
        ground_truth_labels.append(targets)
        
    return prompts_train, prompts_test, ground_truth_labels
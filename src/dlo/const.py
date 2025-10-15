import json
import os

en_aspect_cate_list = ['restaurant general', 'restaurant miscellaneous', 'restaurant prices', 'food quality', 'food prices', 'food style_options', 'drinks quality', 'drinks prices', 'drinks style_options', 'location general', 'service general', 'ambience general']

de_aspect_cate_list = ['Restaurant Allgemein', 'Restaurant Sonstiges', 'Restaurant Preise', 'Essen Qualität', 'Essen Preise', 'Essen Stiloptionen', 'Getränke Qualität', 'Getränke Preise', 'Getränke Stiloptionen', 'Lage Allgemein', 'Service Allgemein', 'Ambiente Allgemein']

cs_aspect_cate_list = ['Restaurace Obecné', 'Restaurace Různé', 'Restaurace Ceny', 'Jídlo Kvalita', 'Jídlo Ceny', 'Jídlo Styl', 'Nápoje Kvalita', 'Nápoje Ceny', 'Nápoje Styl', 'Poloha Obecná', 'Služby Obecné', 'Prostředí Obecné']

tr_aspect_cate_list = ['Restoran Genel', 'Restoran Çeşitli', 'Restoran Fiyatlar', 'Yemek Kalite', 'Yemek Fiyatlar', 'Yemek Tarz', 'İçecek Kalite', 'İçecek Fiyatlar', 'İçecek Tarz', 'Konum Genel', 'Hizmet Genel', 'Ortam Genel']

ru_aspect_cate_list = ['Ресторан Общее', 'Ресторан Разное', 'Ресторан Цены', 'Еда Качество', 'Еда Цены', 'Еда Стиль', 'Напитки Качество', 'Напитки Цены', 'Напитки Стиль', 'Расположение Общее', 'Обслуживание Общее', 'Атмосфера Общее']

fr_aspect_cate_list = ['Restaurant Général', 'Restaurant Divers', 'Restaurant Prix', 'Nourriture Qualité', 'Nourriture Prix', 'Nourriture Style', 'Boissons Qualité', 'Boissons Prix', 'Boissons Style', 'Emplacement Général', 'Service Général', 'Ambiance Général']

es_aspect_cate_list = ['Restaurante General', 'Restaurante Varios', 'Restaurante Precios', 'Comida Calidad', 'Comida Precios', 'Comida Estilo', 'Bebidas Calidad', 'Bebidas Precios', 'Bebidas Estilo', 'Ubicación General', 'Servicio General', 'Ambiente General']

nl_aspect_cate_list = ['Restaurant Algemeen', 'Restaurant Diversen', 'Restaurant Prijzen', 'Voedsel Kwaliteit', 'Voedsel Prijzen', 'Voedsel Stijlopties', 'Dranken Kwaliteit', 'Dranken Prijzen', 'Dranken Stijlopties', 'Locatie Algemeen', 'Service Algemeen', 'Sfeer Algemeen']

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

cate_list = {
    "rest-de": en_aspect_cate_list,
    "rest-en": en_aspect_cate_list,
    "rest-nl": en_aspect_cate_list,
    "rest-es": en_aspect_cate_list,
    "rest-fr": en_aspect_cate_list,
    "rest-ru": en_aspect_cate_list,
    "rest-cs": en_aspect_cate_list,
    "rest-tr": en_aspect_cate_list,
}

task_data_list = {
    "tasd": ['rest-de', 'rest-en', 'rest-nl', 'rest-es', 'rest-fr', 'rest-ru', 'rest-cs',  "rest-tr"],
}

force_words = {
    'tasd': {
        "rest-de": en_aspect_cate_list + list(POLARITY_MAPPINGS_POL_TO_TERM['de'].values()) + ['[SSEP]'],
        "rest-en": en_aspect_cate_list + list(POLARITY_MAPPINGS_POL_TO_TERM['en'].values()) + ['[SSEP]'],
        "rest-nl": en_aspect_cate_list + list(POLARITY_MAPPINGS_POL_TO_TERM['nl'].values()) + ['[SSEP]'],
        "rest-es": en_aspect_cate_list + list(POLARITY_MAPPINGS_POL_TO_TERM['es'].values()) + ['[SSEP]'],
        "rest-fr": en_aspect_cate_list + list(POLARITY_MAPPINGS_POL_TO_TERM['fr'].values()) + ['[SSEP]'],
        "rest-ru": en_aspect_cate_list + list(POLARITY_MAPPINGS_POL_TO_TERM['ru'].values()) + ['[SSEP]'],
        "rest-cs": en_aspect_cate_list + list(POLARITY_MAPPINGS_POL_TO_TERM['cs'].values()) + ['[SSEP]'],
        "rest-tr": en_aspect_cate_list + list(POLARITY_MAPPINGS_POL_TO_TERM['tr'].values()) + ['[SSEP]'],
    }
}
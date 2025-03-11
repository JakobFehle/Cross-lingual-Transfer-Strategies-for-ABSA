import json
import os

senttag2opinion = {'pos': 'great', 'neg': 'bad', 'neu': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

opinion2sentword = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}

senttag2opinion_GER = {'pos': 'gut', 'neg': 'schlecht', 'neu': 'ok'}
sentword2opinion_GER = {'positive': 'gut', 'negative': 'schlecht', 'neutral': 'ok'}

rest_aspect_cate_list = [
    'location general', 'food prices', 'food quality', 'food general',
    'ambience general', 'service general', 'restaurant prices',
    'drinks prices', 'restaurant miscellaneous', 'drinks quality',
    'drinks style_options', 'restaurant general', 'food style_options'
]

rest_16_aspect_cate_list = [
    'restaurant general', 'food quality', 'service general', 'restaurant prices', 'restaurant miscellaneous', 'ambience general',
    'food style_options', 'food prices', 'drinks prices', 'drinks quality', 'drinks style_options', 'location general'
]

base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, "force_tokens.json"), 'r') as f:
    force_tokens = json.load(f)

cate_list = {
    "rest-de": rest_aspect_cate_list,
    "rest-en": rest_aspect_cate_list,
    "rest-nl": rest_aspect_cate_list,
    "rest-es": rest_aspect_cate_list,
    "rest-fr": rest_aspect_cate_list,
    "rest-ru": rest_aspect_cate_list,
    "rest-cs": rest_aspect_cate_list,
}

task_data_list = {
    "tasd": ['rest-de', 'rest-en', 'rest-nl', 'rest-es', 'rest-fr', 'rest-ru', 'rest-cs'],
}

force_words = {
    'tasd': {
        "rest-de": rest_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
        "rest-en": rest_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
        "rest-nl": rest_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
        "rest-es": rest_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
        "rest-fr": rest_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
        "rest-ru": rest_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
        "rest-cs": rest_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
    }
}
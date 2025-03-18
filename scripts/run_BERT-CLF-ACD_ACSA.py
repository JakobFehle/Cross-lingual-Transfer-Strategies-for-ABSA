import subprocess
import sys
import os
import pandas as pd
import numpy as np

# Enable HT
BATCH_SIZE = 16
BASE_EPOCHS = 3
MAX_STEPS = 0
RESULTS_PATH = '../results/bert_clf_cased/'
DATA_PATH = '../data/restaurant/'
LEARNING_RATE = 2e-5
SEED = 5

###
# Hyperparameter Validation Phase
###

for SPLIT in [2,3,4,5]:
    EVAL_TYPE = f'eval_{SPLIT}'
    for DATA_SETTING in ['balanced']:
        for LANG_SETTING in ['adapted']:
            for index, (LANGUAGE, MODEL_NAME) in enumerate([['de', 'deepset/gbert-base'],['en', 'google-bert/bert-base-cased'],['nl', 'GroNLP/bert-base-dutch-cased'], ['ru', 'DeepPavlov/rubert-base-cased'], ['cs', 'UWB-AIR/Czert-B-base-cased'], ['fr', 'dbmdz/bert-base-french-europeana-cased'], ['es', 'dccuchile/bert-base-spanish-wwm-cased']] + ([['tr', 'dbmdz/bert-base-turkish-cased']] if DATA_SETTING != 'balanced' else [])):
                for TASK in ['acsa', 'acd']:
                    for BASE_EPOCHS in [30, 35, 40, 45, 50]:
                        for LEARNING_RATE in [5e-5]:
                            for BATCH_SIZE in [16]:
                                if DATA_SETTING == 'balanced' and LANG_SETTING == 'orig':
                                    MODEL_NAME = 'google-bert/bert-base-multilingual-cased'
                
                                # if index%2 == int(sys.argv[1]):
                                command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 ../src/bert_clf/classifier.py \
                                --seed {SEED} \
                                --task {TASK} \
                                --data_setting {DATA_SETTING} \
                                --lang {LANGUAGE} \
                                --lang_setting {LANG_SETTING} \
                                --model_name_or_path {MODEL_NAME} \
                                --learning_rate {LEARNING_RATE} \
                                --per_device_train_batch_size {BATCH_SIZE} \
                                --num_train_epochs {BASE_EPOCHS} \
                                --output_dir {RESULTS_PATH} \
                                --data_path {DATA_PATH} \
                                --eval_type {EVAL_TYPE}"
                                process = subprocess.Popen(command, shell=True)
                                process.wait()
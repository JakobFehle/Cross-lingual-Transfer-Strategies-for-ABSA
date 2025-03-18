import subprocess
import os
import pandas as pd
import numpy as np
import sys

RESULTS_PATH = '../results/paraphrase/'
TASK = 'tasd'
BATCH_SIZE = 16
BASE_EPOCHS = 30
LEARNING_RATE = 3e-4
GRADIENT_STEPS = 1
DATA_PATH = '../data/restaurant'
SEED = 5

###
# Hyperparameter Validation Phase
###

for SPLIT in [2]:
    EVAL_TYPE = f'eval_{SPLIT}'
    for SEED in [5]:
        for DATA_SETTING in ['orig']:
            for LANG_SETTING in ['adapted']:
                for BASE_EPOCHS in [30]:
                    for index, (LANGUAGE, MODEL) in enumerate([['ru', 'ai-forever/ruT5-base'], ['cs', 'google/mt5-base'], ['fr', 't5-base'], ['es', 'vgaraujov/t5-base-spanish']] + ([['tr', 'google/mt5-base']] if DATA_SETTING != 'balanced' else [])):
                        if DATA_SETTING == 'balanced':
                            MODEL = 'google/mt5-base'
                        # if index%2 == int(sys.argv[1]):
                        command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 ../src/paraphrase/classifier.py \
                        --task {TASK} \
                        --data_path {DATA_PATH} \
                        --data_setting {DATA_SETTING} \
                        --lang {LANGUAGE} \
                        --lang_setting {LANG_SETTING} \
                        --learning_rate {LEARNING_RATE} \
                        --per_device_train_batch_size {BATCH_SIZE} \
                        --num_train_epochs {BASE_EPOCHS} \
                        --model_name_or_path {MODEL} \
                        --output_dir {RESULTS_PATH} \
                        --gradient_accumulation_steps {GRADIENT_STEPS} \
                        --seed {SEED} \
                        --eval_type {EVAL_TYPE}"
                        process = subprocess.Popen(command, shell=True)
                        process.wait()

for SPLIT in [1,2]:
    EVAL_TYPE = f'eval_{SPLIT}'
    for SEED in [5]:
        for DATA_SETTING in ['orig']:
            for LANG_SETTING in ['adapted']:
                for BASE_EPOCHS in [15,20,25,30]:
                    for index, (LANGUAGE, MODEL) in enumerate([['cs', 'google/mt5-base']]):
                        if DATA_SETTING == 'balanced':
                            MODEL = 'google/mt5-base'
                        # if index%2 == int(sys.argv[1]):
                        command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 ../src/paraphrase/classifier.py \
                        --task {TASK} \
                        --data_path {DATA_PATH} \
                        --data_setting {DATA_SETTING} \
                        --lang {LANGUAGE} \
                        --lang_setting {LANG_SETTING} \
                        --learning_rate {LEARNING_RATE} \
                        --per_device_train_batch_size {BATCH_SIZE} \
                        --num_train_epochs {BASE_EPOCHS} \
                        --model_name_or_path {MODEL} \
                        --output_dir {RESULTS_PATH} \
                        --gradient_accumulation_steps {GRADIENT_STEPS} \
                        --seed {SEED} \
                        --eval_type {EVAL_TYPE}"
                        process = subprocess.Popen(command, shell=True)
                        process.wait()

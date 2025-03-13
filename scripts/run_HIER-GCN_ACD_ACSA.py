import subprocess
import sys
import os
import pandas as pd
import numpy as np

DATA_DIR = '../data/restaurant/'
MODEL = 'GCN'
OUTPUT_PATH = '../results/hier_gcn/'
BASE_EPOCHS = 5
BATCH_SIZE = 8
SEED = 5

###
# HT
###

TASK = 'acsa'

for SPLIT in [2]:
    EVAL_TYPE = f'eval_{SPLIT}'
    for DATA_SETTING in ['balanced']:
        for LANG_SETTING in ['orig']:
            for BASE_EPOCHS in [45, 50]:
                for LEARNING_RATE in [2e-5, 5e-5]:
                    for LANGUAGE, MODEL_NAME in [['de', 'deepset/gbert-base'],
                                                ['en', 'google-bert/bert-base-cased'],
                                                ['nl', 'GroNLP/bert-base-dutch-cased'], 
                                                ['ru', 'DeepPavlov/rubert-base-cased'], 
                                                ['cs', 'UWB-AIR/Czert-B-base-cased'], 
                                                ['fr', 'dbmdz/bert-base-french-europeana-cased'], 
                                                ['es', 'dccuchile/bert-base-spanish-wwm-cased']] + ([['tr', 'dbmdz/bert-base-turkish-cased']] if DATA_SETTING != 'balanced' else []):
                        
                        if DATA_SETTING == 'balanced':
                            MODEL_NAME = 'google-bert/bert-base-multilingual-cased'

                        SYS_EXEC = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 " if os.name == "posix" else "python " 
                        command = f"{SYS_EXEC}../src/hier_gcn/run_classifier_gcn.py \
                            --seed {SEED} \
                            --task {TASK} \
                            --lang {LANGUAGE} \
                            --lang_setting {LANG_SETTING} \
                            --eval_type {EVAL_TYPE} \
                            --data_setting {DATA_SETTING} \
                            --do_train \
                            --do_eval \
                            --model_name_or_path {MODEL_NAME} \
                            --model_type {MODEL}\
                            --do_lower_case \
                            --data_dir {DATA_DIR} \
                            --max_seq_length 128 \
                            --per_device_train_batch_size {BATCH_SIZE} \
                            --learning_rate {LEARNING_RATE} \
                            --num_train_epochs {BASE_EPOCHS} \
                            --output_dir {OUTPUT_PATH}"
                        process = subprocess.Popen(command, shell=True)
                        process.wait()

for SPLIT in [3]:
    EVAL_TYPE = f'eval_{SPLIT}'
    for DATA_SETTING in ['balanced']:
        for LANG_SETTING in ['orig']:
            for BASE_EPOCHS in [30,35,40, 45, 50]:
                for LEARNING_RATE in [2e-5, 5e-5]:
                    for LANGUAGE, MODEL_NAME in [['de', 'deepset/gbert-base'],
                                                ['en', 'google-bert/bert-base-cased'],
                                                ['nl', 'GroNLP/bert-base-dutch-cased'], 
                                                ['ru', 'DeepPavlov/rubert-base-cased'], 
                                                ['cs', 'UWB-AIR/Czert-B-base-cased'], 
                                                ['fr', 'dbmdz/bert-base-french-europeana-cased'], 
                                                ['es', 'dccuchile/bert-base-spanish-wwm-cased']] + ([['tr', 'dbmdz/bert-base-turkish-cased']] if DATA_SETTING != 'balanced' else []):
                        
                        if DATA_SETTING == 'balanced':
                            MODEL_NAME = 'google-bert/bert-base-multilingual-cased'

                        SYS_EXEC = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 " if os.name == "posix" else "python " 
                        command = f"{SYS_EXEC}../src/hier_gcn/run_classifier_gcn.py \
                            --seed {SEED} \
                            --task {TASK} \
                            --lang {LANGUAGE} \
                            --lang_setting {LANG_SETTING} \
                            --eval_type {EVAL_TYPE} \
                            --data_setting {DATA_SETTING} \
                            --do_train \
                            --do_eval \
                            --model_name_or_path {MODEL_NAME} \
                            --model_type {MODEL}\
                            --do_lower_case \
                            --data_dir {DATA_DIR} \
                            --max_seq_length 128 \
                            --per_device_train_batch_size {BATCH_SIZE} \
                            --learning_rate {LEARNING_RATE} \
                            --num_train_epochs {BASE_EPOCHS} \
                            --output_dir {OUTPUT_PATH}"
                        process = subprocess.Popen(command, shell=True)
                        process.wait()
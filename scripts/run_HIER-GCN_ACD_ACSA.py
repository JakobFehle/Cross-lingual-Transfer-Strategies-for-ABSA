import subprocess
import sys
import os
import pandas as pd
import numpy as np

DATA_DIR = '../data/restaurant/'
MODEL = 'GCN'
OUTPUT_PATH = '../results/hier_gcn/'
BASE_EPOCHS = 20
BATCH_SIZE = 8
SEED = 5

###
# HT
###

EVAL_TYPE = 'test'
DATA_SETTING = 'balanced' # orig, balanced, multi_id, multi_od
LANG_SETTING = 'orig' # orig, adapted

TASK = 'acsa'
LANG_MODEL = [['de', 'deepset/gbert-base'],
              ['en', 'google-bert/bert-base-uncased'],
              ['nl', 'GroNLP/bert-base-dutch-cased'], 
              ['ru', 'DeepPavlov/rubert-base-cased'], 
              ['cs', 'UWB-AIR/Czert-B-base-cased'], 
              ['fr', 'dbmdz/bert-base-french-europeana-cased'], 
              ['es', 'dccuchile/bert-base-spanish-wwm-uncased']]

for DATA_SETTING in ['multi_id', 'multi_od', 'orig']:
    for LANG_SETTING in ['orig']:
        for LANGUAGE, MODEL_NAME in LANG_MODEL:
            MODEL_NAME = 'google-bert/bert-base-uncased'
            if 'multi' in DATA_SETTING:
                BASE_EPOCHS = 10
            else:
                BASE_EPOCHS = 20
            command = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 ../src/hier_gcn/run_classifier_gcn.py \
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
                  --learning_rate 5e-5 \
                  --num_train_epochs {BASE_EPOCHS} \
                  --output_dir {OUTPUT_PATH}"
            process = subprocess.Popen(command, shell=True)
            process.wait()

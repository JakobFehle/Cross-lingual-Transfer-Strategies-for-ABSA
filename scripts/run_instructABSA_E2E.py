import subprocess
import sys
import os
import pandas as pd
import numpy as np

# Enable HT
ORIGINAL_SPLIT = False 
BATCH_SIZE = 8
BASE_EPOCHS = 4
MAX_STEPS = 0

###
# HT
###
LEARNING_RATE = 5e-5
DATA_PATH = '../data/restaurant/'
OUTPUT_PATH = '../results/instructABSA/'
MODEL = 'allenai/tk-instruct-base-def-pos'
MODEL = 'google/mt5-base'

for SETTING in ['balanced', 'orig']:   
    for LANGUAGE, MODEL in [['de', 't5-base'], ['en', 'allenai/tk-instruct-base-def-pos'], ['nl', 'yhavinga/t5-v1.1-base-dutch-cased'], ['ru', 'ai-forever/ruT5-base'], ['cs', 'google/mt5-base'], ['fr'], ['es']]:
        command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 ../src/instructABSA/run_model.py \
                    -mode train -model_checkpoint {MODEL} \
                    -task joint \
                    -output_dir instructABSA/Models \
                    -inst_type 2 \
                    -data_path {DATA_PATH} \
                    -evaluation_strategy no \
                    -learning_rate {LEARNING_RATE} \
                    -per_device_train_batch_size {BATCH_SIZE} \
                    -num_train_epochs {BASE_EPOCHS} \
                    -setting {SETTING} \
                    -lang {LANGUAGE}"
        process = subprocess.Popen(command, shell=True)
        process.wait()
    
        command = f"CUDA_VISIBLE_DEVICES={sys.argv[1]} python3 ../src/instructABSA/run_model.py \
                    -mode eval  \
                    -model_checkpoint {MODEL} \
                    -task joint \
                    -output_dir instructABSA/Models \
                    -inst_type 2 \
                    -data_path {DATA_PATH} \
                    -evaluation_strategy no \
                    -learning_rate {LEARNING_RATE} \
                    -per_device_train_batch_size {BATCH_SIZE} \
                    -num_train_epochs {BASE_EPOCHS} \
                    -setting {SETTING} \
                    -lang {LANGUAGE} \
                    -output_path {OUTPUT_PATH}"
        process = subprocess.Popen(command, shell=True)
        process.wait()
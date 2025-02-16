import subprocess
import os
import pandas as pd
import numpy as np

RESULTS_PATH = '../results/paraphrase/'
MODEL_NAME = "t5-large"
TASK = 'tasd'
BATCH_SIZE = 16
BASE_EPOCHS = 20
MAX_STEPS = 0
LEARNING_RATE = 3e-4
GRADIENT_STEPS = 2
DATA_PATH = '../data/restaurant'

###
# Hyperparameter Validation Phase
###

SETTING = 'balanced'

for LANGUAGE in ['de','en','nl','ru','cs','fr','es']:
    command = f"python3 ../src/paraphrase/baseline_tasd.py --task {TASK} --setting {SETTING} --lang {LANGUAGE} --learning_rate {LEARNING_RATE} --per_device_train_batch_size {BATCH_SIZE} --num_train_epochs {BASE_EPOCHS} --model_name_or_path {MODEL_NAME} --output_dir {RESULTS_PATH} --gradient_accumulation_steps {GRADIENT_STEPS} --data_path {DATA_PATH}"
    process = subprocess.Popen(command, shell=True)
    process.wait()
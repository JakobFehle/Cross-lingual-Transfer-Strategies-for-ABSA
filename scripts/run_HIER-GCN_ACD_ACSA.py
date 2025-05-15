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

for SPLIT in [0]:
    EVAL_TYPE = f'eval_{SPLIT}'
    for DATA_SETTING in ['orig']:
        for LANG_SETTING in ['adapted']:
            for BASE_EPOCHS in [30, 35, 40, 45, 50]:
                for LEARNING_RATE in [2e-5]:
                    for LANGUAGE, MODEL_NAME in [['en', 'google-bert/bert-base-cased'],['es', 'dccuchile/bert-base-spanish-wwm-cased']]:
                        
                        if 'balanced' in DATA_SETTING and LANG_SETTING == 'orig':
                            MODEL_NAME = 'google-bert/bert-base-multilingual-cased'

                        SYS_EXEC = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 " if os.name == "posix" else "python " 
                        command = f"{SYS_EXEC}../src/hier_gcn/run_classifier_gcn.py \
                            --seed {SEED} \
                            --task {TASK} \
                            --lang {LANGUAGE} \
                            --lang_setting {LANG_SETTING} \
                            --eval_type {EVAL_TYPE} \
                            --data_setting {DATA_SETTING} \
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



RESULTS_PATH = '../results/hier_gcn/'
col_names = ['task', 'lang', 'lang_setting', 'eval_type', 'data_setting', 'learning_rate', 'batch_size', 'epoch', 'seed', 'f1-micro', 'f1-macro', 'accuracy']
runs = []

folder_names = [folder for folder in os.listdir(os.path.join(RESULTS_PATH)) if os.path.isdir(os.path.join(RESULTS_PATH, folder)) and folder != '.ipynb_checkpoints']

for folder_name in folder_names:
    try:
        df = pd.read_csv(os.path.join(RESULTS_PATH, folder_name, 'metrics_phrases.tsv'), sep = '\t')
        df = df.set_index(df.columns[0])
        
        cond_parameters = folder_name.split('_')

        if cond_parameters[3] == 'eval':
            cond_parameters[3] = cond_parameters[3] + '_' + cond_parameters[4]
            cond_parameters.pop(4)
        
        cond_parameters.append(df.loc['Micro-AVG', 'f1'])
        cond_parameters.append(df.loc['Macro-AVG', 'f1'])
        cond_parameters.append(df.loc['Micro-AVG', 'accuracy'])
        runs.append(cond_parameters)
    except:
        pass

results_all = pd.DataFrame(runs, columns = col_names)
results_all["f1-micro"] = pd.to_numeric(results_all["f1-micro"], errors="coerce")

EVAL_TYPE = 'test'

for SPLIT in [0]:
    for SEED in [5, 10, 15, 20, 25]:
        for DATA_SETTING in ['balanced']:
            for LANG_SETTING in ['orig']:
                for index, (LANGUAGE, MODEL_NAME) in enumerate([['de', 'deepset/gbert-base'],['en', 'google-bert/bert-base-cased'],['nl', 'GroNLP/bert-base-dutch-cased'], ['ru', 'DeepPavlov/rubert-base-cased'], ['cs', 'UWB-AIR/Czert-B-base-cased'], ['fr', 'dbmdz/bert-base-french-europeana-cased'], ['es', 'dccuchile/bert-base-spanish-wwm-cased']] + ([['tr', 'dbmdz/bert-base-turkish-cased']] if DATA_SETTING != 'balanced' else [])):
                    results_sub = results_all[np.logical_and.reduce([results_all['data_setting'] == 'balanced-b', 
                                                                 results_all['lang_setting'] == LANG_SETTING, 
                                                                 results_all['lang'] == LANGUAGE, 
                                                                 results_all['eval_type'] == 'eval_0'])].sort_values(by = ['f1-micro'], ascending = False)
                    
                    results_sub = results_sub.reset_index()
        
                    print(results_sub.head(3))
                    BASE_EPOCHS = int(results_sub.at[0, 'epoch'])

                    if 'balanced' in DATA_SETTING and LANG_SETTING == 'orig':
                        MODEL = 'google-bert/bert-base-multilingual-cased'
                        
                    SYS_EXEC = f"CUDA_VISIBLE_DEVICES={int(sys.argv[1])} python3 " if os.name == "posix" else "python " 
                    command = f"{SYS_EXEC}../src/hier_gcn/run_classifier_gcn.py \
                            --seed {SEED} \
                            --task {TASK} \
                            --lang {LANGUAGE} \
                            --lang_setting {LANG_SETTING} \
                            --eval_type {EVAL_TYPE} \
                            --data_setting {DATA_SETTING} \
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


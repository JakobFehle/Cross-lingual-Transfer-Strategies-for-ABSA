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

for SPLIT in [0]:
    EVAL_TYPE = f'eval_{SPLIT}'
    for DATA_SETTING, LANG_SETTING in [['balanced', 'adapted'], ['balanced', 'orig'], ['orig', 'adapted']]:
        for index, (LANGUAGE, MODEL_NAME) in enumerate([['de', 'deepset/gbert-base'],
                                                        ['en', 'google-bert/bert-base-cased'],
                                                        ['nl', 'GroNLP/bert-base-dutch-cased'], 
                                                        ['ru', 'DeepPavlov/rubert-base-cased'], 
                                                        ['cs', 'UWB-AIR/Czert-B-base-cased'], 
                                                        ['fr', 'dbmdz/bert-base-french-europeana-cased'], 
                                                        ['es', 'dccuchile/bert-base-spanish-wwm-cased']] + ([
                                                            ['tr', 'dbmdz/bert-base-turkish-cased']] if DATA_SETTING != 'balanced' else [])):
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


RESULTS_PATH = '../results_test/bert_clf/'
col_names = ['task', 'lang', 'lang_setting', 'eval_type', 'data_setting', 'learning_rate', 'batch_size', 'epoch', 'seed', 'f1-micro', 'f1-macro', 'accuracy']
runs = []

# Paraphrase Generation
folder_names = [folder for folder in os.listdir(os.path.join(RESULTS_PATH)) if os.path.isdir(os.path.join(RESULTS_PATH, folder)) and folder != '.ipynb_checkpoints']

for folder_name in folder_names:
    try:
        cond_parameters = folder_name.split('_')

        if cond_parameters[0] == 'acd':
            df = pd.read_csv(os.path.join(RESULTS_PATH, folder_name, 'metrics_asp.tsv'), sep = '\t')
            df = df.set_index(df.columns[0])
        else:
            df = pd.read_csv(os.path.join(RESULTS_PATH, folder_name, 'metrics_asp_pol.tsv'), sep = '\t')
            df = df.set_index(df.columns[0])

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
    for SEED in [5,10,15,20,25]:
        for TASK in ['acd', 'acsa']:
            for DATA_SETTING, LANG_SETTING in [['orig', 'adapted'], ['balanced', 'orig'], ['balanced', 'adapted']]:
                for index, (LANGUAGE, MODEL_NAME) in enumerate([['de', 'deepset/gbert-base'],['en', 'google-bert/bert-base-cased'],['nl', 'GroNLP/bert-base-dutch-cased'], ['ru', 'DeepPavlov/rubert-base-cased'], ['cs', 'UWB-AIR/Czert-B-base-cased'], ['fr', 'dbmdz/bert-base-french-europeana-cased'], ['es', 'dccuchile/bert-base-spanish-wwm-cased']] + ([['tr', 'dbmdz/bert-base-turkish-cased']] if DATA_SETTING != 'balanced' else [])):

                    results_sub = results_all[np.logical_and.reduce([results_all['data_setting'] == f'{DATA_SETTING}-{DATA_SETTING[0]}', 
                                                                results_all['lang_setting'] == LANG_SETTING, 
                                                                results_all['lang'] == LANGUAGE, 
                                                                results_all['task'] == TASK,
                                                                results_all['eval_type'] == 'eval_0'])].sort_values(by = ['f1-micro'], ascending = False)
                    results_sub = results_sub.reset_index()
        
                    print(results_sub.head(3))
                    BASE_EPOCHS = int(results_sub.at[0, 'epoch'])

                    if DATA_SETTING == 'balanced' and LANG_SETTING == 'orig':
                        MODEL_NAME = 'google-bert/bert-base-multilingual-cased'

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
import subprocess
import sys
import os
import pandas as pd
import numpy as np

###
# CV
###

LEARNING_RATE = 2e-4
DATA_PATH = '../data/restaurant/'
OUTPUT_PATH = '../results/llm/'
TASK = 'tasd'
SEED = 5
BATCH_SIZE = 16
MODEL_NAME = "gemma-3-4b-it"
MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 256
MAX_REG = 10
TEMPERATURE = 0.8

for SPLIT in [0]:
    EVAL_TYPE = f'eval_{SPLIT}'
    for SEED in [5]:
        for DATA_SETTING, LANG_SETTING in [['balanced', 'orig']]:
            for BASE_EPOCHS in [10,15,20]:
                
                for LANGUAGE in ['en']:
                        
                    command = [
                        sys.executable,  # The Python interpreter
                        "../src/llm/classifier.py",  # The script to run
                        "--data_path", DATA_PATH,
                        "--model_name_or_path", MODEL_NAME,
                        "--lang", LANGUAGE,
                        "--lang_setting", LANG_SETTING,
                        "--eval_type", EVAL_TYPE,
                        "--data_setting", DATA_SETTING,
                        "--output_dir", OUTPUT_PATH,
                        "--num_train_epochs", str(BASE_EPOCHS),
                        "--task", TASK,
                        "--seed", str(SEED),
                        "--train_batch_size", str(BATCH_SIZE),
                        "--learning_rate", str(LEARNING_RATE),
                        "--eval_batch_size", "16",
                        "--max_seq_length", str(MAX_SEQ_LENGTH),
                        "--max_new_tokens", str(MAX_NEW_TOKENS),
                        "--max_num_regenerations_eval", str(MAX_REG),
                        "--temperature", str(TEMPERATURE)
                    ]

                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
                    env["TOKENIZERS_PARALLELISM"] = 'true'
                    
                    # Run the subprocess
                    process = subprocess.Popen(command, env=env)
                    process.wait()



# RESULTS_PATH = '../results/llm/'
# col_names = ['task', 'lang', 'lang_setting', 'eval_type', 'data_setting', 'learning_rate', 'batch_size', 'epoch', 'seed', 'f1-micro', 'f1-macro', 'accuracy']
# runs = []

# # Paraphrase Generation
# folder_names = [folder for folder in os.listdir(os.path.join(RESULTS_PATH)) if os.path.isdir(os.path.join(RESULTS_PATH, folder)) and folder != '.ipynb_checkpoints']

# for folder_name in folder_names:
#     try:
#         df = pd.read_csv(os.path.join(RESULTS_PATH, folder_name, 'metrics_phrases.tsv'), sep = '\t')
#         df = df.set_index(df.columns[0])
        
#         cond_parameters = folder_name.split('_')

#         if cond_parameters[3] == 'eval':
#             cond_parameters[3] = cond_parameters[3] + '_' + cond_parameters[4]
#             cond_parameters.pop(4)
        
#         cond_parameters.append(df.loc['Micro-AVG', 'f1'])
#         cond_parameters.append(df.loc['Macro-AVG', 'f1'])
#         cond_parameters.append(df.loc['Micro-AVG', 'accuracy'])
#         runs.append(cond_parameters)
#     except:
#         pass

# results_all = pd.DataFrame(runs, columns = col_names)
# results_all["f1-micro"] = pd.to_numeric(results_all["f1-micro"], errors="coerce")

# EVAL_TYPE = 'test'

# for SPLIT in [0]:
#     for SEED in [5,10,15,20,25]:
#         # for DATA_SETTING, LANG_SETTING in [['orig', 'adapted'], ['balanced', 'orig'], ['balanced', 'adapted']]:
#         for DATA_SETTING, LANG_SETTING in [['orig', 'adapted']]:
#             for index, (LANGUAGE, MODEL) in enumerate([['tr', 'google/mt5-base']]):
#                 results_sub = results_all[np.logical_and.reduce([results_all['data_setting'] == f'{DATA_SETTING}-{DATA_SETTING[0]}', 
#                                                              results_all['lang_setting'] == LANG_SETTING, 
#                                                              results_all['lang'] == LANGUAGE, 
#                                                              results_all['eval_type'] == 'eval_0'])].sort_values(by = ['f1-micro'], ascending = False)
#                 results_sub = results_sub.reset_index()
    
#                 print(results_sub.head(3))
#                 BASE_EPOCHS = int(results_sub.at[0, 'epoch'])




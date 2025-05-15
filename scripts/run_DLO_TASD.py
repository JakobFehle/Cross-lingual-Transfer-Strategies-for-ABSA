import subprocess
import sys
import os
import pandas as pd
import numpy as np

###
# CV
###

LEARNING_RATE = 1e-4
DATA_PATH = '../data/restaurant/'
OUTPUT_PATH = '../results_cd/dlo/'
TASK = 'tasd'
SEED = 5
BATCH_SIZE = 16
GRADIENT_STEPS = 1

EVAL_TYPE = 'test'
for SEED in [10, 15, 20, 25]:
    for DATA_SETTING, LANG_SETTING in [['multi_od', 'orig']]:
        for BASE_EPOCHS in [10]:
            # for index, (LANGUAGE, MODEL) in enumerate([['de', 't5-base'], ['en', 't5-base'], ['nl', 'yhavinga/t5-base-dutch'], ['ru', 'ai-forever/ruT5-base'], ['cs', 'google/mt5-base'], ['fr', 't5-base'], ['es', 'vgaraujov/t5-base-spanish']] + ([['tr', 'google/mt5-base']] if DATA_SETTING != 'balanced' else [])):
            for index, (LANGUAGE, MODEL) in enumerate([['es', 'google/mt5-base'], ['ru', 'google/mt5-base']]):
    
                
                if DATA_SETTING == 'balanced' and LANG_SETTING == 'orig':
                    MODEL = 'google/mt5-base'
    
                if MODEL == 'google/mt5-base':
                    GRADIENT_STEPS = 2
                    
                elif LANGUAGE in ['cs', 'tr']: ## OOM
                    GRADIENT_STEPS = 2
                    
                else:
                    GRADIENT_STEPS = 1
                    
                command = [
                    sys.executable,  # The Python interpreter
                    "../src/dlo/classifier.py",  # The script to run
                    "--data_path", DATA_PATH,
                    "--model_name_or_path", MODEL,
                    "--lang", LANGUAGE,
                    "--lang_setting", LANG_SETTING,
                    "--eval_type", EVAL_TYPE,
                    "--data_setting", DATA_SETTING,
                    "--output_dir", OUTPUT_PATH,
                    "--num_train_epochs", str(BASE_EPOCHS),
                    "--task", TASK,
                    "--top_k", "5",
                    "--seed", str(SEED),
                    "--train_batch_size", str(BATCH_SIZE),
                    "--gradient_accumulation_steps", str(GRADIENT_STEPS),
                    "--learning_rate", str(LEARNING_RATE),
                    "--eval_batch_size", "8",
                    "--max_seq_length", "200"
                ]
                
                if 'multi' in DATA_SETTING:
                    command.append("--constrained_decode")
                    
                # if LANGUAGE != 'de':
                #     command.append("--lowercase")
                
                # Add the environment variable as a prefix
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
                env["TOKENIZERS_PARALLELISM"] = 'true'
                
                # Run the subprocess
                process = subprocess.Popen(command, env=env)
                process.wait()



# RESULTS_PATH = '../results/dlo/'
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

#                 if 'balanced' in DATA_SETTING and LANG_SETTING == 'orig':
#                     MODEL = 'google/mt5-base'
#                     GRADIENT_STEPS = 2
                    
#                 elif LANGUAGE in ['cs', 'tr']: ## OOM
#                     GRADIENT_STEPS = 2
                    
#                 else:
#                     GRADIENT_STEPS = 1
                    
#                 command = [
#                     sys.executable,  # The Python interpreter
#                     "../src/dlo/classifier.py",  # The script to run
#                     "--data_path", DATA_PATH,
#                     "--model_name_or_path", MODEL,
#                     "--lang", LANGUAGE,
#                     "--lang_setting", LANG_SETTING,
#                     "--eval_type", EVAL_TYPE,
#                     "--data_setting", DATA_SETTING,
#                     "--output_dir", OUTPUT_PATH,
#                     "--num_train_epochs", str(BASE_EPOCHS),
#                     "--task", TASK,
#                     "--top_k", "5",
#                     "--seed", str(SEED),
#                     "--train_batch_size", str(BATCH_SIZE),
#                     "--gradient_accumulation_steps", str(GRADIENT_STEPS),
#                     "--learning_rate", str(LEARNING_RATE),
#                     "--eval_batch_size", "16",
#                     "--max_seq_length", "200",
#                 ]

#                 if 'multi' in DATA_SETTING:
#                     command.append("--constrained_decode",)

#                 # if LANGUAGE != 'de':
#                 #     command.append("--lowercase")
                
#                 # Add the environment variable as a prefix
#                 env = os.environ.copy()
#                 env["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
#                 env["TOKENIZERS_PARALLELISM"] = 'true'
                
#                 # Run the subprocess
#                 process = subprocess.Popen(command, env=env)
#                 process.wait()



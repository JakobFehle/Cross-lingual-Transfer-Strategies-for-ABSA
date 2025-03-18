import subprocess
import sys
import os


###
# CV
###

LEARNING_RATE = 1e-4
DATA_PATH = '../data/restaurant/'
OUTPUT_PATH = '../results/dlo/'
TASK = 'tasd'
SEED = 5
EPOCHS = 20
BATCH_SIZE = 16
GRADIENT_STEPS = 1

for SPLIT in [0]:
    EVAL_TYPE = f'eval_{SPLIT}'
    for SEED in [5]:
        for DATA_SETTING in ['balanced']:
            for LANG_SETTING in ['adapted']:
                for BASE_EPOCHS in [15, 20, 25, 30]:
                    for index, (LANGUAGE, MODEL) in enumerate([['de', 't5-base'], ['en', 't5-base'], ['nl', 'yhavinga/t5-v1.1-base-dutch-uncased'], ['ru', 'ai-forever/ruT5-base'], ['cs', 'google/mt5-base'], ['fr', 't5-base'], ['es', 'vgaraujov/t5-base-spanish']] + ([['tr', 'google/mt5-base']] if DATA_SETTING != 'balanced' else [])):
                        if DATA_SETTING == 'balanced' and LANG_SETTING == 'orig':
                            MODEL = 'google/mt5-base'
                        if LANGUAGE == 'cs': ## OOM
                            BATCH_SIZE = 8
                            GRADIENT_STEPS = 2
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
                            "--eval_batch_size", "16",
                            "--max_seq_length", "200",
                            "--do_train"
                        ]

                        # if LANGUAGE != 'de':
                        #     command.append("--lowercase")
                        
                        # Add the environment variable as a prefix
                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
                        # env["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
                        
                        # Run the subprocess
                        process = subprocess.Popen(command, env=env)
                        process.wait()





# for SEED in [5, 10, 15, 20, 25]:
#     for DATA_SETTING in ['balanced']:
#         for LANG_SETTING in ['orig']:
#             # for index, (LANGUAGE, MODEL) in enumerate([['de', 'google/mt5-base'], ['en', 't5-base'], ['nl', 'yhavinga/t5-v1.1-base-dutch-cased'], ['ru', 'ai-forever/ruT5-base'], ['cs', 'google/mt5-base'], ['fr', 'google/mt5-base'], ['es', 'google/mt5-base']] + ([['tr', '']] if DATA_SETTING != 'balanced' else [])):
#             for index, (LANGUAGE, MODEL) in enumerate([['de', 'google/mt5-base'], ['en', 't5-base'], ['nl', 'yhavinga/t5-v1.1-base-dutch-cased'], ['ru', 'ai-forever/ruT5-base'], ['cs', 'google/mt5-base'], ['fr', 'google/mt5-base'], ['es', 'google/mt5-base']]):
#                 if DATA_SETTING == 'balanced':
#                     MODEL = 't5-base'
#                 if 'multi' in DATA_SETTING:
#                     BASE_EPOCHS = 10
#                 else:
#                     BASE_EPOCHS = 20
#                 # if index%2 == int(sys.argv[1]):
#                 command = [
#                     sys.executable,  # The Python interpreter
#                     "../src/ilo/classifier.py",  # The script to run
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
#                     "--gradient_accumulation_steps", "1",
#                     "--learning_rate", str(LEARNING_RATE),
#                     "--eval_batch_size", "16",
#                     "--lowercase",
#                     "--max_seq_length", "200",
#                     "--do_train"
#                 ]
        
#                 # if LANGUAGE != 'de':
#                 #     command.append("--lowercase")
                
#                 # Add the environment variable as a prefix
#                 env = os.environ.copy()
#                 env["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
#                 # env["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
                
#                 # Run the subprocess
#                 process = subprocess.Popen(command, env=env)
#                 process.wait()

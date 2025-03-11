import subprocess
import sys
import os


###
# CV
###

LEARNING_RATE = 3e-4
DATA_PATH = '../data/restaurant/'
OUTPUT_PATH = '../results/mvp/'
TASK = 'tasd'
SEED = 42
EPOCHS = 20
BATCH_SIZE = 16
GRADIENT_STEPS = 2

for SEED in [5]:
    for DATA_SETTING in ['orig']:
        for LANG_SETTING in ['orig']:
            for BASE_EPOCHS in [20,25,30,35,40]:
            # for index, (LANGUAGE, MODEL) in enumerate([['de', 'google/mt5-base'], ['en', 't5-base'], ['nl', 'yhavinga/t5-v1.1-base-dutch-cased'], ['ru', 'ai-forever/ruT5-base'], ['cs', 'google/mt5-base'], ['fr', 'google/mt5-base'], ['es', 'google/mt5-base']] + ([['tr', '']] if DATA_SETTING != 'balanced' else [])):
            # for index, LANGUAGE in enumerate(['de', 'en', 'nl', 'ru', 'cs', 'fr', 'es'] + (['tr'] if DATA_SETTING != 'balanced' else [])):
                MODEL = 'google/mt5-base'
                LANGUAGE = 'en'
                
                # if index%2 == int(sys.argv[1]):
                command = [
                    sys.executable,  # The Python interpreter
                    "../src/mvp/src/classifier.py",  # The script to run
                    "--data_path", DATA_PATH,
                    "--model_name_or_path", MODEL,
                    "--lang", LANGUAGE,
                    "--lang_setting", LANG_SETTING,
                    "--eval_type", EVAL_TYPE,
                    "--data_setting", DATA_SETTING,
                    "--output_dir", OUTPUT_PATH,
                    "--num_train_epochs", str(BASE_EPOCHS),
                    "--save_top_k", "0",
                    "--task", TASK,
                    "--top_k", "5",
                    "--ctrl_token", "post",
                    "--multi_path",
                    "--num_path", "5",
                    "--seed", str(SEED),
                    "--train_batch_size", str(BATCH_SIZE),
                    "--gradient_accumulation_steps", str(GRADIENT_STEPS),
                    "--learning_rate", str(LEARNING_RATE),
                    "--sort_label",
                    "--data_ratio", "1.0",
                    "--check_val_every_n_epoch", str(BASE_EPOCHS),
                    "--agg_strategy", "vote",
                    "--eval_batch_size", "16",
                    "--constrained_decode",
                    "--do_train",
                    "--lowercase"
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

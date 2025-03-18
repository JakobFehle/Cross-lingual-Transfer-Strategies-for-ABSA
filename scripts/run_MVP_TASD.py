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
EPOCHS = 2
BATCH_SIZE = 16
GRADIENT_STEPS = 1

for SPLIT in [0]:
    EVAL_TYPE = f'eval_{SPLIT}'
    for SEED in [5]:
        for DATA_SETTING in ['balanced']:
            for LANG_SETTING in ['adapted']:
                for BASE_EPOCHS in [5, 10]:
                    for index, (LANGUAGE, MODEL) in enumerate([
                        ['nl', 'yhavinga/t5-base-dutch', 4e-5], 
                        ['ru', 'ai-forever/ruT5-base', 1e-5], 
                        ['cs', 'google/mt5-base'], 
                        ['de', 't5-base'], 
                        ['en', 't5-base'], 
                        ['fr', 't5-base'], 
                        ['es', 'vgaraujov/t5-base-spanish']] + ([
                        ['tr', 'google/mt5-base']] if DATA_SETTING != 'balanced' else [])):
                        if DATA_SETTING == 'balanced' and LANG_SETTING == 'orig':
                            MODEL = 'google/mt5-base'
                            BATCH_SIZE = 16
                            GRADIENT_STEPS = 2

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
                            "--check_val_every_n_epoch", str(BASE_EPOCHS+1),
                            "--agg_strategy", "vote",
                            "--eval_batch_size", "16",
                            "--constrained_decode",
                            "--lowercase"
                        ]
                
                        # if LANGUAGE != 'de':
                        #     command.append("--lowercase")
                        
                        # Add the environment variable as a prefix
                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
                        env["TOKENIZERS_PARALLELISM"] = 'false'
                        # env["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
                        
                        # Run the subprocess
                        process = subprocess.Popen(command, env=env)
                        process.wait()

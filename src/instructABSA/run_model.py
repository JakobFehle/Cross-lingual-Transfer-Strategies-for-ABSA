import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import torch
from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator, T5Classifier
from InstructABSA.config import Config
from instructions import InstructionsHandler

try:
    use_mps = True if torch.has_mps else False
except:
    use_mps = False

# Set Global Values
config = Config()
instruct_handler = InstructionsHandler()
if config.inst_type == 1:
    instruct_handler.load_instruction_set1()
else:
    instruct_handler.load_instruction_set2()
    
if config.mode == 'train':
    if config.data_path is None:
        raise Exception('Please provide training data path for mode=training.')
    
if config.mode == 'eval':
    if config.data_path is None:
        raise Exception('Please provide testing data path for mode=eval.')
         

config.experiment_name = f'e2e_{config.lang}_{config.setting}_{config.learning_rate}_{config.per_device_train_batch_size}_{str(int(config.num_train_epochs))}'

if config.mode == 'train':
    print('Experiment Name: ', config.experiment_name)
    model_checkpoint = config.model_checkpoint
    model_out_path = config.output_dir
    model_out_path = os.path.join(model_out_path, config.task, f"{model_checkpoint.replace('/', '')}-{config.experiment_name}")
else:
    model_checkpoint = os.path.join(config.output_dir, config.task, f"{config.model_checkpoint.replace('/', '')}-{config.experiment_name}")
    model_out_path = config.output_dir

print('Mode set to: ', 'training' if config.mode == 'train' else ('inference' if config.mode == 'eval' \
                                                                  else 'Individual sample inference'))

def loadDataset(data_path, lang, setting):
    def transformData(labels):
        labels_new = []
        for label in labels:
            sentiment = f"{label[1]}"
            phrase = label[2] if label[2] != "NULL" else "noaspectterm"
            phrase = f"'{phrase}'" if not "'" in phrase else f'"{phrase}"'
            labels_new.append(f"{{'term': {phrase}, 'polarity': '{sentiment}'}}")
        labels_new = ', '.join(labels_new)
        return f'[{labels_new}]'
        
    df_train = pd.read_json(os.path.join(data_path, lang, f'train{"_b" if setting == "balanced" else ""}.json'), orient="records", lines=True).set_index('id')
    df_test = pd.read_json(os.path.join(data_path, lang, f'test{"_b" if setting == "balanced" else ""}.json'), orient="records", lines=True).set_index('id')

    df_train['labels'] = df_train['labels'].apply(transformData)
    df_test['labels'] = df_test['labels'].apply(transformData)

    df_train.columns = ['raw_text', 'aspectTerms']
    df_test.columns = ['raw_text', 'aspectTerms']

    print(df_train.head(10))
    print(df_test.head(10))
    return df_train, df_test

if config.mode != 'cli':
    if config.data_path is not None:
        train_df, test_df = loadDataset(config.data_path, config.lang, config.setting)
        ood_tr_df, ood_te_df = None, None
    print('Loaded data...')
else:
    print('Running inference on input: ', config.test_input)

# Training arguments
training_args = {
                'output_dir': model_out_path,
                'evaluation_strategy': config.evaluation_strategy if config.data_path is not None else 'no',
                'learning_rate': config.learning_rate,
                'per_device_train_batch_size': config.per_device_train_batch_size if config.per_device_train_batch_size is not None else None,
                'per_device_eval_batch_size': config.per_device_eval_batch_size,
                'num_train_epochs': config.num_train_epochs if config.num_train_epochs is not None else None,
                'weight_decay': config.weight_decay,
                'warmup_ratio': config.warmup_ratio,
                'save_strategy': config.save_strategy,
                'load_best_model_at_end': config.load_best_model_at_end,
                'push_to_hub': config.push_to_hub,
                'eval_accumulation_steps': config.eval_accumulation_steps,
                'predict_with_generate': config.predict_with_generate,
                'use_mps_device': use_mps
            }

# Create T5 model object
print(config.set_instruction_key)
if config.set_instruction_key == 1:
    indomain = 'bos_instruct1'
    outdomain = 'bos_instruct2'
else:
    indomain = 'bos_instruct2'
    outdomain = 'bos_instruct1'

if config.task == 'ate':
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.ate[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.ate[outdomain]
    eos_instruction = instruct_handler.ate['eos_instruct']
if config.task == 'atsc':
    t5_exp = T5Classifier(model_checkpoint)
    bos_instruction_id = instruct_handler.atsc[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.atsc[outdomain]
    delim_instruction = instruct_handler.atsc['delim_instruct']
    eos_instruction = instruct_handler.atsc['eos_instruct']
if config.task == 'joint':
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.aspe[indomain]
    eos_instruction = instruct_handler.aspe['eos_instruct']

if config.mode != 'cli':
    # Define function to load datasets and tokenize datasets
    loader = DatasetLoader(train_df, test_df, None, ood_tr_df, ood_te_df, None, config.sample_size)
    if config.task == 'ate':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_ate_format(loader.train_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_ate_format(loader.test_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_ate_format(loader.train_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_ate_format(loader.test_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)

    elif config.task == 'atsc':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_atsc_format(loader.train_df_ood, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_atsc_format(loader.test_df_ood, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)

    elif config.task == 'joint':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_aspe_format(loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_aspe_format(loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        # if loader.train_df_ood is not None:
        #     loader.train_df_ood = loader.create_data_in_aspe_format(loader.train_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
        # if loader.test_df_ood is not None:
        #     loader.test_df_ood = loader.create_data_in_aspe_format(loader.test_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)

    # Tokenize dataset
    id_ds, id_tokenized_ds, ood_ds, ood_tokenized_ds = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs) 

    if config.mode == 'train':
        # Train model
        model_trainer = t5_exp.train(id_tokenized_ds, **training_args)
        print('Model saved at: ', model_out_path)
    elif config.mode == 'eval':
        # Get prediction labels
        print('Model loaded from: ', model_checkpoint)
        if id_tokenized_ds.get("train") is not None:
            id_tr_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'train', 
                                                  batch_size=config.per_device_eval_batch_size, 
                                                  max_length = config.max_token_length)
            id_tr_df = pd.DataFrame(id_ds['train'])[['text', 'labels']]
            id_tr_df['labels'] = id_tr_df['labels'].apply(lambda x: x.strip())
            id_tr_df['pred_labels'] = id_tr_pred_labels
            os.makedirs(os.path.join(config.output_path, config.experiment_name), exist_ok=True)
            id_tr_df.to_csv(os.path.join(config.output_path, config.experiment_name, 'id_train.csv'), index=False)
            print('*****Train Metrics*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(id_tr_df['labels'], id_tr_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', recall)
            print('F1-Score: ', f1)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)


        if id_tokenized_ds.get("test") is not None:
            id_te_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'test', 
                                                  batch_size=config.per_device_eval_batch_size, 
                                                  max_length = config.max_token_length)
            id_te_df = pd.DataFrame(id_ds['test'])[['text', 'labels']]
            id_te_df['labels'] = id_te_df['labels'].apply(lambda x: x.strip())
            id_te_df['pred_labels'] = id_te_pred_labels
            os.makedirs(os.path.join(config.output_path, config.experiment_name), exist_ok=True)
            id_te_df.to_csv(os.path.join(config.output_path, config.experiment_name, 'id_test.csv'), index=False)
            print('*****Test Metrics*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(id_te_df['labels'], id_te_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', recall)
            print('F1-Score: ', f1)
            
            with open(os.path.join(config.output_path, config.experiment_name, 'results.tsv'), "w") as writer:
                for key, value in {'Precision': precision, 'Recall': recall, 'F1-Score': f1}.items():
                    writer.write("%s\t%s\n" % (key, value))
                
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)

        if ood_tokenized_ds.get("train") is not None:
            ood_tr_pred_labels = t5_exp.get_labels(tokenized_dataset = ood_tokenized_ds, sample_set = 'train', 
                                                   batch_size=config.per_device_eval_batch_size, 
                                                   max_length = config.max_token_length)
            ood_tr_df = pd.DataFrame(ood_ds['train'])[['text', 'labels']]
            ood_tr_df['labels'] = ood_tr_df['labels'].apply(lambda x: x.strip())
            ood_tr_df['pred_labels'] = ood_tr_pred_labels
            os.makedirs(os.path.join(config.output_path, config.experiment_name), exist_ok=True)
            ood_tr_df.to_csv(os.path.join(config.output_path, config.experiment_name, 'ood_train.csv'), index=False)
            print('*****Train Metrics - OOD*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(ood_tr_df['labels'], ood_tr_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', precision)
            print('F1-Score: ', precision)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)
            
        if ood_tokenized_ds.get("test") is not None:
            ood_te_pred_labels = t5_exp.get_labels(tokenized_dataset = ood_tokenized_ds, sample_set = 'test', 
                                                   batch_size=config.per_device_eval_batch_size, 
                                                   max_length = config.max_token_length)
            ood_te_df = pd.DataFrame(ood_ds['test'])[['text', 'labels']]
            ood_te_df['labels'] = ood_te_df['labels'].apply(lambda x: x.strip())
            ood_te_df['pred_labels'] = ood_te_pred_labels
            os.makedirs(os.path.join(config.output_path, config.experiment_name), exist_ok=True)
            ood_te_df.to_csv(os.path.join(config.output_path, config.experiment_name, 'ood_test.csv'), index=False)
            print('*****Test Metrics - OOD*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(ood_te_df['labels'], ood_te_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', precision)
            print('F1-Score: ', precision)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)
else:
    print('Model loaded from: ', model_checkpoint)
    if config.task == 'atsc':
        config.test_input, aspect_term = config.test_input.split('|')[0], config.test_input.split('|')[1]
        model_input = bos_instruction_id + config.test_input + f'. The aspect term is: {aspect_term}' + eos_instruction
    else:
        model_input = bos_instruction_id + config.test_input + eos_instruction
    input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
    outputs = t5_exp.model.generate(input_ids, max_length = config.max_token_length)
    print('Model output: ', t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True))

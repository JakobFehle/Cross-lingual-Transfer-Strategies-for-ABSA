import unsloth
import os, re, sys, json, random, torch, time, argparse

utils = os.path.abspath('../src/utils/') # Relative path to utils scripts
sys.path.append(utils)

from tqdm import tqdm
from preprocessing import loadDataset
import numpy as np
import torch, subprocess, json
from datasets import Dataset
from prompts import *
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported, FastModel

from evaluation import createResults, convertLabels
from trl import SFTTrainer

from validator import validate_label, to_pred_list

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision != 0 or recall != 0
        else 0
    )
    scores = {"precision": precision, "recall": recall, "f1": f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, task, label_space):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):

        gold_list = extract_spans_para(task, gold_seqs[i], "gold")
        pred_list = extract_spans_para(task, pred_seqs[i], "pred")

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    scores = compute_f1_scores(all_preds, all_labels)

    preds = [
        [f"{lbl[1]}:{lbl[0]}:{lbl[2]}"
         for lbl in pred if f"{lbl[1]}::{lbl[0]}" != "::"]
        for pred in all_preds
    ]

    golds = [
        [f"{lbl[1]}:{lbl[0]}:{lbl[2]}" for lbl in gold if f"{lbl[1]}::{lbl[2]}" != "::"]
        for gold in all_labels
    ]

    scores_dfs = createResults(preds, golds, label_space, task)
    print('LLM F1-Micro Scuffed: ', scores['f1'])
    return scores_dfs, all_labels, all_preds


def get_prompt_header(language):
    return globals()[f'PROMPT_{language.upper()}']

def get_model_and_tokenizer(max_seq_length, model_name_or_path, seed):
    dtype = None
    load_in_4bit = True
    load_in_8bit = False
    model, tokenizer = FastModel.from_pretrained(
        model_name=f"{model_name_or_path}",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Turn off for just text!
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # SHould leave on always!
        r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=8,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=seed,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    return model, tokenizer


def get_trainer(model, tokenizer, dataset, args):
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=args.train_batch_size,
            warmup_steps=0,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
    )
    
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data/", type=str)
    parser.add_argument("--task", default='asqp', type=str)
    parser.add_argument("--model_name_or_path", default='t5-base', type=str)
    parser.add_argument("--output_dir", default='outputs/temp', type=str)
    parser.add_argument("--lang", default='en', choices=["de", "en", "nl", "ru", "cs", "fr", "es", "tr"], type=str)
    parser.add_argument("--eval_type", default='test', choices=["test", "eval_0"], type=str)
    parser.add_argument("--data_setting", default="orig", choices=["orig", "balanced", "multi_id", "multi_od"], type=str)
    parser.add_argument("--lang_setting", default="orig", choices=["orig", "adapted"],  type=str)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--max_new_tokens", default=200, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    # parser.add_argument("--cond_name", default='', type=str)
    parser.add_argument("--max_model_len", default=2048, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    args = parser.parse_args()

    return args


def extract_output_as_list(output_raw, task):
    try:
        output_label = output_raw.split("### Label:")[1]
        output_label = output_label.split("]")[0] + "]"

        n_elements = {"asqp": 4, "tasd": 3, "e2e": 2}
        output_label = to_pred_list(output_label, n_elements[task])
        output_label = [
            _tuple for _tuple in output_label if len(_tuple) == n_elements[task]
        ]
    except:
        output_label = []
    return output_label

def createPrompts(dataset, test = False, eos_token = ''):
    def formatting_prompts_func(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for input, output in zip(inputs, outputs):
            text = PROMPT_TEMPLATE.format(prompt_header, input, output) + eos_token
            texts.append(text)
        return {
            "text": texts,
        }

    unique_aspect_categories = sorted(set([asp.split(':')[0] for asp in label_space]))
        
    prompt_header = (
        get_prompt_header(lanugage)
        .replace("[[aspect_category]]", str(unique_aspect_categories)[1:-1])
        .replace("[[examples]]", "")[:-2]
    )

    if test:
        predictions = {}

        all_prompts = []
        all_labels = []
    
        for idx, example in tqdm(test_ds.iterrows(), total=test_ds.shape[0]):
            test_text = example["text"]
            all_prompts.append(
                PROMPT_TEMPLATE.format(
                    prompt_header,
                    test_text,
                    "",
                )
            )
    
            tuple_list = [list(_tuple) for _tuple in [tuple(examples) for examples in example["labels"]]]
            tuple_list = [[_tuple[0], _tuple[1], *_tuple[2:]] for _tuple in tuple_list]
            all_labels.append(tuple_list)

        return all_prompts, all_labels

    else:
        examples_preprocessed = []
        for idx, examples in train_ds.iterrows():
            examples_preprocessed.append(
                {
                    "input": example["text"],
                    "output": str([tuple(s.strip() for s in labels) for labels in examples["labels"]]),
                }
            )
        examples_preprocessed = Dataset.from_dict({key: [d[key] for d in examples_preprocessed] for key in examples_preprocessed[0]})
    
        all_prompts = examples_preprocessed.map(
            formatting_prompts_func,
            batched=True,
        )

        return all_prompts
        
    

if __name__ == '__main__':
    args = init_args()
    
    set_seed(args.seed)

    model, tokenizer = get_model_and_tokenizer(args.max_seq_length, args.model_name_or_path, args.seed)

    if args.data_setting == 'balanced':
        train, test_balanced, label_space = splitForEvalSetting(loadDataset(args.data_path, args.lang, args.data_setting), args.eval_type)
        _, test_orig, _ = loadDataset(args.data_path, args.lang, 'orig')

        unique_aspect_categories = sorted(set([asp.split(':')[0] for asp in label_space]))
        
        prompt_header = (
            get_prompt_header(args.lang)
            .replace("[[aspect_category]]", str(unique_aspect_categories)[1:-1])
            .replace("[[examples]]", "")[:-2]
        )

        if 'eval' in args.eval_type:
            test_dataset, gold = createPrompts(test_balanced, test = True, eos_token = tokenizer.eos_token)

    elif args.data_setting == 'orig':
        train, test_orig, label_space = splitForEvalSetting(loadDataset(args.data_path, args.lang, args.data_setting), args.eval_type)
        if args.lang != 'tr':
            _, test_balanced, _ = loadDataset(args.data_path, args.lang, 'balanced')

        unique_aspect_categories = sorted(set([asp.split(':')[0] for asp in label_space]))
        
        prompt_header = (
            get_prompt_header(args.lang)
            .replace("[[aspect_category]]", str(unique_aspect_categories)[1:-1])
            .replace("[[examples]]", "")[:-2]
        )
        
        if 'eval' in args.eval_type:
            test_dataset, gold = createPrompts(test_orig, test = True, eos_token = tokenizer.eos_token)

    else: 
        train, test_balanced, label_space = splitForEvalSetting(loadDataset(args.data_path, args.lang, 'multi_balanced'), args.eval_type)
        _, test_orig, _ = loadDataset(args.data_path, args.lang, 'orig')

    train_dataset = createPrompts(train, test = False, eos_token = tokenizer.eos_token)
        
    if train:
        model, tokenizer = get_model_and_tokenizer(args.max_seq_length, model_name_or_path, seed)
        
        start_total_time = time.time()
    
        trainer = get_trainer(model, tokenizer, train_dataset, args)
        trainer.train()
        
        end_total_time = time.time()
        training_duration = end_total_time - start_total_time

        model_config = f"../src/llm/model_cache/{args.task}_{args.lang}_{args.lang_setting}_{args.eval_type}_{args.data_setting}_{args.learning_rate}_{args.batch_size}_{args.num_train_epochs}_{args.seed}"
        model.save_pretrained(model_config, maximum_memory_usage=0.9)

        trainer_args = {}
        trainer_args.update({
            "model_name": args.model_name_or_path,
            "task": args.task,
            "data_setting": args.data_setting,
            "lang": args.lang,
            "lang_setting": args.lang_setting,
            "per_device_train_batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "eval_type": args.eval_type,
            "train_runtime": training_duration
        })
    else:
        total_time = 0


    ###
    #  Eval
    ###
    
    if 'eval' not in args.eval_type:
        if args.lang != 'tr':
            test_dataset_balanced, gold_balanced = createPrompts(test_balanced, test = True, eos_token = tokenizer.eos_token)
            res_config_1 = f"../results/llm/{args.task}_{args.lang}_{args.lang_setting}_{args.eval_type}_{args.data_setting}-b_{args.learning_rate}_{args.batch_size}_{args.num_train_epochs}_{args.seed}/"
            os.makedirs(res_config_1, exist_ok=True)
            
            with open(f"{res_config_1}res_config.json", "w") as f:
                json.dump({"all_prompts": test_dataset_balanced, "all_labels": gold_balanced, "train_config": trainer_args, "label_space": label_space}, f)
            
        test_dataset_orig, gold_orig = createPrompts(test_orig, test = True, eos_token = tokenizer.eos_token)
        res_config_2 = f"../results/llm/{args.task}_{args.lang}_{args.lang_setting}_{args.eval_type}_{args.data_setting}-o_{args.learning_rate}_{args.batch_size}_{args.num_train_epochs}_{args.seed}/"
        os.makedirs(res_config_2, exist_ok=True)
        
        with open(f"{res_config_2}res_config.json", "w") as f:
            json.dump({"all_prompts": test_dataset_orig, "all_labels": gold_orig, "train_config": trainer_args, "label_space": label_space}, f)

    else:
        res_config_1 = f"../results/llm/{args.task}_{args.lang}_{args.lang_setting}_{args.eval_type}_{args.data_setting}-{args.data_setting[0]}_{args.learning_rate}_{args.batch_size}_{args.num_train_epochs}_{args.seed}/"
        os.makedirs(res_config_1, exist_ok=True)
        res_config_2 = None
        
        with open(f"{res_config_1}res_config.json", "w") as f:
            json.dump({"all_prompts": test_dataset, "all_labels": gold, "train_config": trainer_args, "label_space": label_space}, f)

    cmd = [
            "python", "../src/llm/test_llm.py",
            "--max_new_tokens", str(args.max_new_tokens), 
            "--max_model_len", str(args.max_model_len),
            "--model_name_or_path", args.model_name_or_path,
            "--seed", str(args.seed),
            "--temperature", str(args.temperature),
            "--evaluation_data_1", res_config_1,
            "--model_path", model_config,
            "--task", args.task,
        ]

    if res_config_2:
        cmd.extend(["--evaluation_data_2", res_config_2])
        
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        # stderr=sys.stdout
    )
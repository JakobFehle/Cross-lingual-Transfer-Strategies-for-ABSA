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
    parser.add_argument("--max_num_regenerations_eval", default=10, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    # parser.add_argument("--cond_name", default='', type=str)
    parser.add_argument("--max_model_len", default=2048, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    args = parser.parse_args()

    return args
    
def savePredictions(f1, golds, preds, args, config, eval_type):
    output_path = os.path.join(args.output_dir, f'{args.task}_{args.lang}_{args.lang_setting}_{args.eval_type}_{args.data_setting.replace("_","-")}-{"b" if eval_type == "balanced" else "o"}_{args.learning_rate}_{args.train_batch_size}_{args.num_train_epochs}_{args.seed}')
    
    os.makedirs(output_path, exist_ok=True)

    for idx, name in enumerate(["asp", "asp_pol", "pairs", "pol", "phrases"]):
        pd.DataFrame.from_dict(f1[idx]).transpose().to_csv(os.path.join(output_path, f"metrics_{name}.tsv"), sep="\t")
    
    try:
        matched_samples = [
            {"predictions": pred, "gold_labels": gold}
            for pred, gold in zip(preds, golds)
        ]
        print(matched_samples[:5])
        with open(os.path.join(output_path, 'predictions.json'), "w", encoding="utf-8") as f:
            json.dump({"test": matched_samples}, f, indent=4, ensure_ascii=False)

    except:
        pass

    with open(os.path.join(output_path, 'config.json'), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

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

if __name__ == '__main__':
    args = init_args()
    
    seed = args.seed
    model_name_or_path = args.model_name_or_path
    task = args.task
    lanugage = args.lang
    max_seq_length = args.max_seq_length
    max_new_tokens = args.max_new_tokens
    max_num_regenerations_eval = args.max_num_regenerations_eval
    temperature = args.temperature
    max_model_len = args.max_model_len

    set_seed(seed)

    train_ds, test_ds, label_space = loadDataset(args.data_path, args.lang, args.data_setting)
    
    unique_aspect_categories = sorted(set([asp.split(':')[0] for asp in label_space]))

    prompt_header = (
        get_prompt_header(lanugage)
        .replace("[[aspect_category]]", str(unique_aspect_categories)[1:-1])
        .replace("[[examples]]", "")[:-2]
    )

    cache_config = f"cache_res_{args.task}_{args.lang}_{args.eval_type}_{args.data_setting}_{args.lang_setting}_{args.num_train_epochs}"
    
    if False:
        dataset = []
        for idx, example in train_ds.iterrows():
            dataset.append(
                {
                    "input": example["text"],
                    "output": str([tuple(s.strip() for s in examples) for examples in example["labels"]]),
                }
            )
        dataset = Dataset.from_dict({key: [d[key] for d in dataset] for key in dataset[0]})
        model, tokenizer = get_model_and_tokenizer(max_seq_length, model_name_or_path, seed)
        EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    
        def formatting_prompts_func(examples):
            inputs = examples["input"]
            outputs = examples["output"]
            texts = []
            for input, output in zip(inputs, outputs):
                # Must add EOS_TOKEN, otherwise your generation will go on forever!
                text = PROMPT_TEMPLATE.format(prompt_header, input, output) + EOS_TOKEN
                texts.append(text)
            return {
                "text": texts,
            }
    
        dataset = dataset.map(
            formatting_prompts_func,
            batched=True,
        )
        
        print(dataset[0])
        
        start_total_time = time.time()
    
        trainer = get_trainer(model, tokenizer, dataset, args)
        trainer.train()
        
        end_total_time = time.time()
        total_time = end_total_time - start_total_time

        model.save_pretrained(cache_config, maximum_memory_usage=0.9)
    else:
        total_time = 0

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

    print(all_prompts[0])
    # save cache_res

    
    with open(f"{cache_config}.json", "w") as f:
        json.dump({"all_prompts": all_prompts, "all_labels": all_labels}, f)

    result = subprocess.run(
        [
            "python", "../src/llm/test_llm.py",
            "--max-new-tokens", str(max_new_tokens), 
            "--max-model-len", str(max_model_len),
            "--model-name-or-path", model_name_or_path,
            "--seed", str(seed),
            "--temperature", str(temperature),
            "--evaluation_data", f"{cache_config}.json",
            "--model_path", cache_config,
            "--task", task,
        ],
        capture_output=True,
        text=True,
        # stderr=sys.stdout
    )
    try:
        result = json.loads(result.stdout.split("#######\n")[1])
    except:
        print(result.stdout)

    predictions["all_prompts"] = all_prompts
    predictions["all_preds"] = result["all_preds"]
    predictions["all_labels"] = all_labels
    total_time = end_total_time - start_total_time
    predictions["total_time"] = total_time
    predictions["scores"] = result["scores"]

    scores_dfs = compute_scores(result["all_preds"], all_labels, task, label_space):

    savePredictions(scores_dfs, golds, preds, args, config, eval_type):
    
    print(predictions["scores"])
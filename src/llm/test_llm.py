from vllm import LLM
import torch, json, argparse
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
import json, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath('../src/utils/'))

from evaluation import createResults, convertLabels
from validator import to_pred_list
from transformers import set_seed
import random
import numpy as np
import torch
import pandas as pd

parser = argparse.ArgumentParser(description="Run LLM model")
parser.add_argument(
    "--model_name_or_path",
    "-m",
    type=str,
    default="gemma-3-4b-it",
    help="Model name or path",
)
parser.add_argument("--seed", "-s", type=int, default=0, help="Random seed")
parser.add_argument("--evaluation_data_1", "-e1", type=str, help="Path to evaluation data")
parser.add_argument("--evaluation_data_2", "-e2", type=str, default=None, help="Path to evaluation data")
parser.add_argument(
    "--temperature",
    "-t",
    type=float,
    default=0,
    help="Temperature for sampling",
)

parser.add_argument(
    "--max_model_len",
    "-ml",
    type=int,
    default=4096,
    help="Maximum model length",
)

parser.add_argument(
    "--max_new_tokens",
    "-mnt",
    type=int,
    default=1024,
    help="Maximum new tokens to generate",
)

parser.add_argument(
    "--task",
    "-ta",
    type=str,
    default="asqp",
    help="Task to run",
)

parser.add_argument(
    "--model_path",
    "-mp",
    type=str,
    help="Path to the model",
)

def savePredictions(f1, golds, preds, output_path, training_args):
    
    os.makedirs(output_path, exist_ok=True)

    for idx, name in enumerate(["asp", "asp_pol", "pairs", "pol", "phrases"]):
        if f1[idx] is not None:
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
        json.dump(training_args, f, indent=4, ensure_ascii=False)

def extract_output_as_list(output_raw, task):
    try:
        output_label = output_raw

        n_elements = {"asqp": 4, "tasd": 3, "acsa": 2, "acd": 1}
        output_label = to_pred_list(output_label, n_elements[task])
        output_label = [
            list(_tuple) for _tuple in output_label if len(_tuple) == n_elements[task]
        ]
        # switch first and second element of each tuple
        if task == 'tasd':
            output_label = [[_tuple[1], _tuple[2].upper(), _tuple[0]] for _tuple in output_label]
        elif task == 'acsa':
            output_label = [[_tuple[0], _tuple[1].upper()] for _tuple in output_label]
        elif task == 'acd':
            output_label = [_tuple for _tuple in output_label]
    except:
        output_label = []
    return output_label

def createSamplingParams(config):
    STOP_WORDS = ["### Input:", "\n\n", "Sentence:"]

    return SamplingParams(
        temperature=config["temperature"],
        stop=STOP_WORDS,
        max_tokens=config["max_new_tokens"],
        top_k=config["top_k"],
        top_p=config["top_p"],
        skip_special_tokens=True,
        seed=config['seed']
    )
   
def compute_f1_scores(pred_pt, gold_pt):
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1
                
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision != 0 or recall != 0
        else 0
    )
    scores = {"precision": precision, "recall": recall, "f1": f1}

    return scores

args = parser.parse_args()

set_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Load the model
model = LLM(
    model=args.model_name_or_path,
    tokenizer=args.model_name_or_path,
    dtype="bfloat16",
    max_model_len=args.max_model_len,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization = 0.6,
    seed=args.seed,
    enable_lora=True,
    max_lora_rank=64,
    max_num_seqs = 20
)

# Generate predictions
eval_config = {
    "temperature": args.temperature,
    "max_new_tokens": args.max_new_tokens,
    "seed": args.seed,
    "top_k": -1,
    "top_p": 1,
}

sampling_params = createSamplingParams(eval_config)

eval_paths = [args.evaluation_data_1]
if args.evaluation_data_2:
    eval_paths.append(args.evaluation_data_2)


for path in eval_paths:
    
    # load the evaluation data (json)
    with open(f"{path}res_config.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    all_prompts = eval_data["all_prompts"]
    all_labels = eval_data["all_labels"]
    label_space = eval_data["label_space"]
    training_args = eval_data["train_config"]
    
    model_outputs = model.generate(all_prompts, sampling_params, lora_request=LoRARequest("adapter", 1, args.model_path))

    all_preds = []
    
    for k in range(len(model_outputs)):
       all_preds.append(extract_output_as_list(model_outputs[k].outputs[0].text, args.task))
    
    for k in range(len(all_labels)):
       all_labels[k] = [
           list(_tuple) for _tuple in all_labels[k]
       ] 

    print(all_preds[0])
    print(all_labels[0])
    
    scores = compute_f1_scores(all_preds, all_labels)
    print("\nF1 Scores - Nils: ", scores['f1'])

    preds, _ = convertLabels(all_preds, args.task, label_space)
    golds, _ = convertLabels(all_labels, args.task, label_space)

    print(preds[0])
    print(golds[0])
    
    f1 = createResults(preds, golds, label_space, args.task)
    if args.task == 'tasd':
        print("\nF1 Scores - Jakob: ", f1[4]['Micro-AVG']['f1'])
    elif args.task == 'acsa':
        print("\nF1 Scores - Jakob: ", f1[1]['Micro-AVG']['f1'])
    elif args.task == 'acd':
        print("\nF1 Scores - Jakob: ", f1[0]['Micro-AVG']['f1'])
    
    training_args.update(eval_config)
    
    savePredictions(f1, all_labels, all_preds, path, training_args)
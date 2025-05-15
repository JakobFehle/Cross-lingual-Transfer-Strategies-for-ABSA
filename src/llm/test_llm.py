from vllm import LLM
import torch, json, argparse
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
import json, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from validator import to_pred_list
from transformers import set_seed
import random
import numpy as np
import torch

parser = argparse.ArgumentParser(description="Run LLM model")
parser.add_argument(
    "--model-name-or-path",
    "-m",
    type=str,
    default="gemma-3-4b-it",
    help="Model name or path",
)
parser.add_argument("--seed", "-s", type=int, default=0, help="Random seed")
parser.add_argument("--evaluation_data", "-e", type=str, help="Path to evaluation data")

parser.add_argument(
    "--temperature",
    "-t",
    type=float,
    default=0,
    help="Temperature for sampling",
)

parser.add_argument(
    "--max-model-len",
    "-ml",
    type=int,
    default=4096,
    help="Maximum model length",
)

parser.add_argument(
    "--max-new-tokens",
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

args = parser.parse_args()

model_name_or_path = args.model_name_or_path
seed = args.seed
evaluation_data_path = args.evaluation_data
temperature = args.temperature
max_model_len = args.max_model_len
max_new_tokens = args.max_new_tokens
task = args.task
model_path = args.model_path

set_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
# load the evaluation data (json)
with open(evaluation_data_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

all_prompts = eval_data["all_prompts"]
all_labels = eval_data["all_labels"]

# Load the model
model = LLM(
    model=model_name_or_path,
    tokenizer=model_name_or_path,
    dtype="bfloat16",
    max_model_len=max_model_len,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization = 0.9,
    seed=seed,
    enable_lora=True,
    max_lora_rank=8,
    max_num_seqs = 20
)

# Generate predictions
config = {
    "temperature": temperature,
    "max_new_tokens": max_new_tokens,
    "seed": seed,
    "top_k": -1,
    "top_p": 1,
}


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



print('\n\n\nStart Inference\n\n\n')
sampling_params = createSamplingParams(config)
model_outputs = model.generate(all_prompts, sampling_params, lora_request=LoRARequest("adapter", 1, model_path))

all_preds = []

def extract_output_as_list(output_raw, task):
    try:
        output_label = output_raw

        n_elements = {"asqp": 4, "tasd": 3, "e2e": 2}
        output_label = to_pred_list(output_label, n_elements[task])
        output_label = [
            list(_tuple) for _tuple in output_label if len(_tuple) == n_elements[task]
        ]
        # switch first and second element of each tuple
        output_label = [[_tuple[1], _tuple[0], *_tuple[2:]] for _tuple in output_label]
    except:
        output_label = []
    return output_label
 
for k in range(len(model_outputs)):
   all_preds.append(extract_output_as_list(model_outputs[k].outputs[0].text, task))

for k in range(len(all_labels)):
   all_labels[k] = [
       list(_tuple) for _tuple in all_labels[k]
   ] 
   
   
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

scores = compute_f1_scores(all_preds, all_labels)

print("#######")
print(json.dumps({"scores": scores, "all_preds": all_preds}))
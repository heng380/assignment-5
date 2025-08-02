import torch
from grpo import *
from argparse import ArgumentParser
import wandb 
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
from vllm import LLM, SamplingParams
import random
import json
from cs336_alignment.utils import tokenize_prompt_and_output, get_response_log_probs, masked_normalize, compute_entropy, sft_microbatch_train_step
from drgrpo_grader import r1_zero_reward_fn
from typing import Callable, List, Tuple
import re
from baseline import run_vllm
from transformers import PreTrainedTokenizerBase
import torch.nn as nn

n_grpo_steps = 200
learning_rate = 1e-5
advantage_eps = 1e-6
rollout_batch_size = 256
group_size = 8
sampling_temperature = 1.0
sampling_min_tokens = 4
sampling_max_tokens = 1024
epochs_per_rollout_batch = 1   # on policy
train_batch_size = 256
gradient_accumulation_steps = 128 # microbatch=2
gpu_memory_utilization = 0.85
loss_type = "reinforce_with_baseline"
use_std_normalization = True

QWEN_MATH_BASE_PATH = "/home/aiscuser/repos/assignment5-alignment/data/model/Qwen2.5-Math-1.5B"
PROMPT_PATH = "/home/aiscuser/repos/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
TEST_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/data/gsm8k/test.jsonl"
OUTPUT_PATH = "/home/aiscuser/repos/assignment5-alignment/data/grpo"
MATH_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/data/gsm8k/train.jsonl"
SEED = 69
torch.manual_seed(SEED)
random.seed(SEED)
device_train = "cuda:3"
device_vllm = "cuda:1"



def train_grpo():
    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    wandb.init(
        project="cs336-grpo",
        name=f"reinforce_with_baseline",
        config={
            "n_grpo_steps": n_grpo_steps
            }
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=QWEN_MATH_BASE_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_train
    )

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MATH_BASE_PATH)
    
    vllm = init_vllm(QWEN_MATH_BASE_PATH, device_vllm, SEED, gpu_memory_utilization=0.9)

    optimizer = torch.optim.AdamW(model.parameter(), lr=learning_rate, weight_decay=0, betas=(0.9, 0.95))

    train_data, test_data = prepare_train_test()  # [{prompt, answer}]

    for grpo_step in range(n_grpo_steps):
        rollout_dataset = random.sample(train_data, n_prompts_per_rollout_batch)



def load_jsonl(file_path:str)->list[str]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_qa_prompt(data:list[str], prompt_path:str)->list[str]:
    formated_q = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    for d in data:
        pair = {}
        pair["prompt"] = prompt.format(question = d["question"])
        pair["answer"] = d["answer"]
        formated_q.append(pair)
    return formated_q

def prepare_train_test():
    train_data = load_jsonl(MATH_DATA_PATH)
    test_data = load_jsonl(TEST_DATA_PATH)
    train_data = format_qa_prompt(train_data, PROMPT_PATH)
    test_data = format_qa_prompt(test_data, PROMPT_PATH)
    return train_data, test_data   # [{prompt, answer}, ...]









def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float=0.85):
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization
        )

if __name__ == "__main__":
    train_grpo()

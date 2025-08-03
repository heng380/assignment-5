import torch
from grpo import *
from argparse import ArgumentParser
import wandb 
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
cliprange = 0.2
grpo_eval_freq = 8
grpo_num_eval_samples = 1024

QWEN_MATH_BASE_PATH = "/home/aiscuser/repos/assignment5/data/model/Qwen2.5-Math-1.5B"
PROMPT_PATH = "/home/aiscuser/repos/assignment5/cs336_alignment/prompts/r1_zero.prompt"
TEST_DATA_PATH = "/home/aiscuser/repos/assignment5/data/gsm8k/test.jsonl"
OUTPUT_PATH = "/home/aiscuser/repos/assignment5/data/grpo"
MATH_DATA_PATH = "/home/aiscuser/repos/assignment5/data/gsm8k/train.jsonl"
SEED = 69
torch.manual_seed(SEED)
random.seed(SEED)
device_train = "cuda:3"
device_vllm = "cuda:1"

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")

def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"

def train_grpo():
    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    # wandb.init(
    #     project="cs336-grpo",
    #     name=f"reinforce_with_baseline",
    #     config={
    #         "n_grpo_steps": n_grpo_steps
    #         }
    # )
    wandb_step = 0

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
        rollout_prompts = [data["prompt"] for data in rollout_dataset]
        rollout_answers = [data["answer"] for data in rollout_dataset]
        
        sampling_params = SamplingParams(temperature = sampling_temperature,
                                        top_p = 1.0, max_tokens=sampling_max_tokens,
                                        min_tokens=sampling_min_tokens,
                                        stop=["</answer>"],
                                        include_stop_str_in_output=True,
                                        n=group_size, seed=SEED)
        outputs = vllm.generate(rollout_prompts, sampling_params)

        repeated_answers = []
        responses = []
        prompts = []
        for output, answer in zip(outputs, rollout_answers):
            answer = extract_reference_answer(answer)
            prompt = output.prompt
            for rollout in output.outputs:
                responses.append(rollout.text)
                prompts.append(prompt)
                repeated_answers.append(answer)
        tokenizations = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids, labels, response_mask = tokenizations["input_ids"], tokenizations["labels"], tokenizations["response_mask"]
        
        advantages_train, raw_rewards_train, metadata = compute_group_normalized_reward(r1_zero_reward_fn, 
                                                                                        rollout_responses=responses, 
                                                                                        repeated_ground_truths=repeated_answers,
                                                                                        group_size=group_size,
                                                                                        advantage_eps=advantage_eps, 
                                                                                        normalized_by_std=use_std_normalization,
                                                                                        )
        print ("---------examples of prompt, response, answer-----------")
        for i in range(3):
            print (f"prompt:{prompts[i]}")
            print (f"rollouts:{responses[i]}")
            print (f"answers:{repeated_answers[i]}")
            print (f"reward:{raw_rewards_train[i]}")
            print ()
        print ("--------grpo step rollout example done")

        num_train_steps_per_epoch = rollout_batch_size // train_batch_size #rollout total num

        # get per token log probs on old model
        with torch.no_grad():
            old_log_probs_train = []
            for train_step in range(num_train_steps_per_epoch):
                batch_idxs = train_step*train_batch_size, (train_step+1)*train_batch_size
                for train_microstep in range(gradient_accumulation_steps):
                    microbatch_idxs = batch_idxs[0] + train_microstep*micro_train_batch_size, batch_idxs[0] + (train_microstep+1)*micro_train_batch_size
                    input_id_micro_batch = input_ids[microbatch_idxs[0]:microbatch_idxs[1]]
                    labels_micro_batch = labels[microbatch_idxs[0]:microbatch_idxs[1]]
                    response_mask_micro_batch = response_mask[microbatch_idxs[0]:microbatch_idxs[1]]
                    log_probs_dict = get_response_log_probs(model=model,
                                                            input_ids=input_id_micro_batch,
                                                            labels = labels_micro_batch,
                                                            return_token_entropy=True)
                    log_probs = log_probs_dict["log_probs"]
                    token_entropy = log_probs_dict["token_entropy"]
                    old_log_probs_train.append(log_probs)
                    assert log_probs.shape[0] == microbatch_idxs[1] - microbatch_idxs[0]
            old_log_probs_train = torch.cat(old_log_probs_train)
        print (f"grpo step {grpo_step}: complete computing log probs on the old model, old_log_probs_train.shape={old_log_probs_train.shape}")


        # train new model
        for train_epoch in range(epochs_per_rollout_batch):
            for train_step in range(num_train_steps_per_epoch):
                batch_idxs = train_step*train_batch_size, (train_step+1)*train_batch_size
                for train_microstep in range(gradient_accumulation_steps):
                    microbatch_idxs = batch_idxs[0] + train_microstep*micro_train_batch_size, batch_idxs[0] + (train_microstep+1)*micro_train_batch_size
                    raw_rewards = raw_rewards_train[microbatch_idxs[0]:microbatch_idxs[1]].unsqueeze(-1)
                    advantages = advantages_train[microbatch_idxs[0]:microbatch_idxs[1]].unsqueeze(-1)
                    old_log_probs = old_log_probs_train[microbatch_idxs[0]:microbatch_idxs[1]]
                    input_id_micro_batch = input_ids[microbatch_idxs[0]:microbatch_idxs[1]]
                    labels_micro_batch = labels[microbatch_idxs[0]:microbatch_idxs[1]]
                    response_mask_micro_batch = response_mask[microbatch_idxs[0]:microbatch_idxs[1]]

                    log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                    log_probs = log_probs_dict["log_probs"]
                    token_entropy = log_probs_dict["token_entropy"]

                    policy_log_probs = log_probs
                    loss, metadata = grpo_microbatch_train_step(policy_log_probs, response_mask_micro_batch, gradient_accumulation_steps, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
                    print (f"train: grpo step {grpo_step}, train epoch {train_epoch}, train step {train_step}, micro batch step {train_microstep}, loss is {loss:.6f}")

                    avg_token_entropy = masked_mean(token_entropy, response_mask, dim=None)
                    wandb.log({
                        "train/train_loss": loss, 
                        "train/train_entropy": avg_token_entropy, 
                        "train_step": wandb_step
                    })
                    if loss_type == "grpo_clip":
                        clipped_fraction = masked_mean(metadata["clipped"], response_mask, dim=None)
                        wandb.log({"train/clip_fraction": clipped_fraction}, step=wandb_step)
                wandb_step += 1
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                optimizer.zero_grad()

        load_policy_into_vllm_instance(model, vllm)
        if grpo_step % grpo_eval_freq == 0:
            prompts = [data["prompt"] for data in test_data]
            answers = [data["answer"] for data in test_data]
            sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)
            overview = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, answers, sampling_params)

            wandb.log({
                "eval/correct": overview["correct"],
                "eval/correct format with wrong answer": overview["answer_wrong"],
                "eval/wrong format": overview["format_wrong"],
                "eval/accuracy": overview["correct"] / overview["count"],
                "eval_step": wandb_step+1
            })

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams
):
    responses = run_vllm(vllm_model, prompts, eval_sampling_params)
    allinfo_dict_list = []
    for response, answer, prompt in zip(responses, answers, prompts):
        extracted_answer = extract_reference_answer(answer)
        reward_dict = reward_fn(response, extracted_answer)
        allinfo_dict_list.append(reward_dict)
    overview = {"correct":0, "format_wrong":0, "answer_wrong":0, "count":0}
    for reward in allinfo_dict_list:
        overview["count"] += 1
        if reward["reward"] == 1:
            overview["correct"] += 1
        elif reward["format_reward"] == 1:
            overview["answer_wrong"] += 1
        else:
            overview["format_wrong"] += 1
    return overview
    
        


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm:LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

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

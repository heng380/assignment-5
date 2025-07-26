import torch
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

QWEN_MATH_BASE_PATH = "/home/aiscuser/repos/assignment5-alignment/data/model/Qwen2.5-Math-1.5B"
PROMPT_PATH = "/home/aiscuser/repos/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
TEST_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/data/gsm8k/test.jsonl"
OUTPUT_PATH = "/home/aiscuser/repos/assignment5-alignment/data/sft/"
MATH_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/cs336_alignment/baseline_result.jsonl"
SEED = 69
torch.manual_seed(SEED)
random.seed(SEED)
device_train = "cuda:2"
device_vllm = "cuda"
micro_batch_size=2
n_sft_steps = 64
n_grad_accum_steps = 32
eval_steps = 8

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")

def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"

def format_qa(data:list[str])->list[str]:
    formated_qa = []
    for d in data:
        pair = {}
        pair["prompt"] = d["prompt"]
        pair["response"] = d["response"]
        formated_qa.append(pair)
    return formated_qa

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

def load_jsonl(file_path:str)->list[str]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


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

def load_policy_into_vllm_instance(policy: torch.nn.Module, llm:LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def prepare_train_test(train_sample)->tuple[list[str], list[str]]:
    train_data = load_jsonl(MATH_DATA_PATH)
    train_data = train_data[:train_sample]
    train_data = format_qa(train_data)

    test_data = load_jsonl(TEST_DATA_PATH)
    test_data = format_qa_prompt(test_data, PROMPT_PATH)
    return train_data, test_data

def get_batch(tokenized_train_data: dict[str, torch.Tensor], batch_size: int, device: str):
    batch_indices = random.sample(range(len(tokenized_train_data["input_ids"])), batch_size)
    return {k: v[batch_indices].to(device) for k,v in tokenize_prompt_and_output}

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
        count += 1
        if reward["reward"] == 1:
            overview["correct"] += 1
        elif reward["format_reward"] == 1:
            overview["answer_wrong"] += 1
        else:
            overview["format_wrong"] += 1
    return overview

def main(train_samples:list[int], dataset_type:str, MATH_DATA_PATH:str) -> None:
    for train_sample in train_samples:
        wandb.init(project="cs336-sft",
            name=f"train_sample_{train_sample}_dataset_{dataset_type}_math_sft",
            configs={
                "train_sample": train_sample,
                "dataset_type": dataset_type
                }
            )
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=QWEN_MATH_BASE_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_train
        )
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MATH_BASE_PATH)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        amp_ctx = torch.amp.autocast(device_type=device_train, dtype=torch.bfloat16)

        vllm = init_vllm(QWEN_MATH_BASE_PATH, device_vllm, SEED, gpu_memory_utilization=0.9)

        train_qa, test_prompt = prepare_train_test(train_sample)

        tokenized_train_data = tokenize_prompt_and_output(prompt_strs=[data["prompt"] for data in train_qa],
                                                          output_strs=[data["response"] for data in train_qa],
                                                          tokenizer=tokenizer)
        train_batch = get_batch(tokenized_train_data, micro_batch_size, device_train)

        input_ids = train_batch["input_ids"].to(device_train)
        labels = train_batch["labels"].to(device_train)
        response_mask = train_batch["response_mask"].to(device_train)
        for i_sft_step in range(n_sft_steps):
            for j_grad_accum_step in range(n_grad_accum_steps):
                with amp_ctx:
                    response_log_probs = get_response_log_probs(model, input_ids, labels, True)
                    log_probs = response_log_probs["log_probs"]
                    entropy = response_log_probs["token_entropy"]

                    next_batch = get_batch(tokenized_train_data, micro_batch_size, device_train)

                    loss, _ = sft_microbatch_train_step(log_probs, response_mask, n_grad_accum_steps)
                    input_ids = train_batch["input_ids"].to(device_train)
                    labels = train_batch["labels"].to(device_train)
                    response_mask = train_batch["response_mask"].to(device_train)

                    if i_sft_step % eval_steps == 0:
                        load_policy_into_vllm_instance(model, vllm)
                        sampling_params = SamplingParams(
                            temperature = 1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_step_str_in_output=True
                        )
                        overview = evaluate_vllm(
                            vllm_model=vllm,
                            reward_fn=r1_zero_reward_fn,
                            prompts = [data["prompt"] for data in test_prompt],
                            answers = [data["answer"] for data in test_prompt],
                            eval_sampling_params = sampling_params
                        )
                        accuracy = overview["correct"] / overview["count"]
                        print (f"evaluation at step {i_sft_step}")











if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--test_type")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--use_correct", type=bool, default=False)

    args = parser.parse_args()
    # for test
    if args.command == "test":
        if args.test_type == "load_data":
            train, test = prepare_train_test(10)
            print (train[0])
            print ("-------")
            print (test[0])

    # for true run
    if args.command == "train":
        if args.use_correct == False:
            MATH_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/cs336_alignment/baseline_result.jsonl"
            train_samples = [128, 256, 512, 1024, 7473]
            dataset_type = "raw"
        else:
            MATH_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/data/gsm8k/train_correct.jsonl"
            train_samples = [215]
            dataset_type = "correct"
    
        main(train_samples, dataset_type, MATH_DATA_PATH)
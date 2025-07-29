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
from transformers import PreTrainedTokenizerBase
import torch.nn as nn

QWEN_MATH_BASE_PATH = "/home/aiscuser/repos/assignment5-alignment/data/model/Qwen2.5-Math-1.5B"
PROMPT_PATH = "/home/aiscuser/repos/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
TEST_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/data/gsm8k/test.jsonl"
OUTPUT_PATH = "/home/aiscuser/repos/assignment5-alignment/data/ei"
MATH_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/data/gsm8k/processed_train.jsonl"
SEED = 69
torch.manual_seed(SEED)
random.seed(SEED)
device_train = "cuda:3"
device_vllm = "cuda:1"
micro_batch_size=8
n_sft_steps = 256
n_grad_accum_steps = 8
eval_steps = 16

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
        pair["answer"] = d["response"]
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

def prepare_train_test()->tuple[list[str], list[str]]:
    train_data = load_jsonl(MATH_DATA_PATH)
    train_data = format_qa(train_data)

    test_data = load_jsonl(TEST_DATA_PATH)
    test_data = format_qa_prompt(test_data, PROMPT_PATH)
    return train_data, test_data

def get_train_inf_batch(prompts: list[str], train_qa: list[dict[str, str]], batch_size: int) -> tuple[list[str], list[dict[str, str]]]: 
    """
    sample batch_size from train prompts for inference, and return the sampled train_qa
    """
    batch_indices = random.sample(range(len(prompts)), batch_size)
    return [prompts[i] for i in batch_indices], [train_qa[i] for i in batch_indices]

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
def to_float(val):
    if isinstance(val, torch.Tensor):
        return val.float().item()
    return float(val)
def get_ei_sft_batch(tokenized_train_data, batch_size, device):
    batch_indices = random.sample(range(len(tokenized_train_data["input_ids"]), batch_size))
    return {k: v[batch_indices] for k, v in tokenized_train_data.items()}


def ei_sft(sft_data: list[dict[str, str]], model:torch.nn.Module, tokenizer:PreTrainedTokenizerBase, vllm:torch.nn.Module, epoch:int, ):
    tokenized_train_data = tokenize_prompt_and_output(prompt_strs=[data["prompt"] for data in sft_data], 
                                                      output_strs=[data["response"] for data in sft_data],
                                                      tokenizer=tokenizer)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
    
    amp_ctx = torch.amp.autocast(device_type=device_train, dtype=torch.bfloat16)
    n_sft_steps = len(sft_data) * epoch // (n_grad_accum_steps * micro_batch_size)

    for i in range(n_sft_steps):
        for j in range(n_grad_accum_steps):
            with amp_ctx:
                train_batch = get_ei_sft_batch(tokenized_train_data, micro_batch_size)
                input_ids = train_batch["input_ids"]
                labels = train_batch["labels"]
                response_mask = train_batch["response_mask"]
                response_log_probs = get_response_log_probs(model, input_ids, labels, True)
                loss, _ = sft_microbatch_train_step(response_log_probs, response_mask, n_grad_accum_steps)
                if j == n_grad_accum_steps - 1:
                    nn.util


    

def main(n_ei_steps:int, batch_size:int, epochs:int, ei_num_g:int) -> None:
    wandb.init(project="cs336-ei-sft",
        name=f"step_{n_ei_steps}_batch_size_{batch_size}_epochs_{epochs}_math_ei_sft",
        config={
            "n_ei_steps": n_ei_steps,
            "batch_size": batch_size,
            "epochs": epochs
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

    vllm = init_vllm(QWEN_MATH_BASE_PATH, device_vllm, SEED, gpu_memory_utilization=0.9)

    train_qa, test_qa = prepare_train_test()

    for i in range(n_ei_steps):
        print (f"Expert iteration step: {n_ei_steps}")
        train_batch = get_train_inf_batch(prompts = [i["prompt"] for i in train_qa], train_qa=train_qa, batch_size=batch_size)

        sampling_params = SamplingParams(temperature = 1.0, top_p=1.0, max_tokens=1024, min_tokens=4, stop=["</answer>"], include_stop_str_in_output=True, n=ei_num_g)
        outputs = vllm.generate(train_batch, sampling_params)
        responses = [[o.text.strip() for o in output.outputs] for output in outputs]
        expert_roll = []
        overview = {"total": 0, "correct": 0, "format_wrong": 0, "format_correct_answer_wrong": 0}

        for response, train_data in zip(responses, train_batch):
            answer = train_data["answer"]
            extracted_answer = extract_reference_answer(answer)

            for rollout in response:
                overview["total"] += 1
                metrics = r1_zero_reward_fn(rollout, extracted_answer)
                if metrics["reward"] > 0:
                    
                    overview["correct"] += 1
                    expert_roll.append({
                        "prompt": train_data["prompt"],
                        "response": rollout
                    })
                elif metrics["format_reward"] > 0:
                    overview["format_correct_answer_wrong"] += 1
                else:
                    overview["format_wrong"] += 1
    
        print (f"correct examples:")
        print (f"prompt:{expert_roll[0]["prompt"]}")
        print (f"response: {expert_roll[0]["response"]}")

        print (f"correction check:")
        print (f"acccuracy: {overview["correct"]/overview["total"] * 100:.2f}%")
        print (f"format correct but answer wrong: {overview["format_correct_answer_wrong"]/overview["total"] * 100:.2f}%")
        print (f"answer wrong: {overview["answer_wrong"]/overview['total'] * 100:.2f}%")

        sft_data = expert_roll
        ei_sft(sft_data, model, tokenizer, vllm)

        load_policy_into_vllm_instance(model, vllm)

if __name__ == "__main__":
    parser = ArgumentParser()
    # subparsers = parser.add_subparsers(dest="command")

    # test_parser = subparsers.add_parser("test")
    # test_parser.add_argument("--test_type")

    # train_parser = subparsers.add_parser("train")
    parser.add_argument("--n_ei_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--ei_num_g", type=int, default=5)
    args = parser.parse_args()
    n_ei_steps = args.n_ei_steps
    batch_size = parser.batch_size
    epochs = parser.epochs
    main(n_ei_steps, batch_size, epochs, ei_num_g)
    # # for test
    # if args.command == "test":
    #     if args.test_type == "load_data":
    #         train, test = prepare_train_test(10)
    #         print (train[0])
    #         print ("-------")
    #         print (test[0])

    # # for true run
    # if args.command == "train":
    #     if args.use_correct == False:
            
    #         train_samples = [7000]
    #         dataset_type = "raw"
    #     else:
    #         MATH_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/data/gsm8k/correct.jsonl"
    #         n_ei_steps = args.n_ei_steps
import torch
from argparse import ArgmumentParser
import wandb 
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from vllm import LLM

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


QWEN_MATH_BASE_PATH = "/home/aiscuser/repos/assignment5-alignment/data/model/Qwen2.5-Math-1.5B"
PROMPT_PATH = "/home/aiscuser/repos/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
TEST_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/data/gsm8k/test.jsonl"

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




if __name__ == "__main__":
    parser = ArgmumentParser()
    parser.add_argument("--use_correct", type=bool, default = False)
    args = parser.parse_args()

    if args.use_correct:
        MATH_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/data/gsm8k/train.jsonl"
        train_samples = [128, 256, 512, 1024]
        dataset_type = "raw"
    else:
        MATH_DATA_PATH = "/home/aiscuser/repos/assignment5-alignment/data/gsm8k/train_correct.jsonl"
        train_samples = [215]
        dataset_type = "correct"
    
    main(train_samples, dataset_type, MATH_DATA_PATH)
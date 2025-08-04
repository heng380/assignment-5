from vllm import LLM, SamplingParams
from typing import Callable, List, Tuple
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import re
import json
import os
from collections import Counter
import argparse
#%%
QWEN_MATH_BASE_PATH = "/home/aiscuser/repos/assignment-5/data/model/Qwen2.5-Math-1.5B"
PROMPT_PATH = "/home/aiscuser/repos/assignment-5/cs336_alignment/prompts/r1_zero.prompt"
MATH_DATA_PATH = "/home/aiscuser/repos/assignment-5/data/gsm8k"

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")


def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"

def run_vllm(vllm_model, prompts, sampling_params) -> List[str]:
    result = vllm_model.generate(prompts, sampling_params)
    texts = [output.outputs[0].text.strip() for output in result]
    return texts

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
        reward_dict["response"] = response
        reward_dict["answer"] = answer
        reward_dict["prompt"] = prompt
        reward_dict["extracted_answer"] = extracted_answer
        allinfo_dict_list.append(reward_dict)
    return allinfo_dict_list
#%%

def load_and_format_prompts(data_path: str, prompt_path: str):
    with open(prompt_path, "r") as file:
        prompt = file.read()
    prompts = []
    answers = []
    with open(data_path, "r") as file:
        for line in file:
            data = json.loads(line)
            prompts.append(prompt.format(question=data["question"]))
            answers.append(data["answer"])
    return prompts, answers

def build_llm_and_params(model_path: str) -> Tuple[LLM, SamplingParams]:
    llm = LLM(model_path)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    return llm, sampling_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--choice")
    args = parser.parse_args()

    if args.choice == "quick_inf":
        ## example for inference
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
            ]
        sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
        )
        
        llm = LLM(model=QWEN_MATH_BASE_PATH, trust_remote_code=True)

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text.strip()
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        ## end of example
    
    if args.choice == "load_prompt_answer":
        prompts, answers = load_and_format_prompts(data_path=MATH_DATA_PATH+"/test.jsonl", prompt_path=PROMPT_PATH)
        for i,j in zip(prompts, answers):
            print (f"prompt:{i}, \n answer:{j}")
            break
    else:
        prompts, answers = load_and_format_prompts(data_path=MATH_DATA_PATH+"/train.jsonl", prompt_path=PROMPT_PATH)
        llm, sampling_params = build_llm_and_params(QWEN_MATH_BASE_PATH)
        allinfo_dict_list = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params)
        with open("baseline_result.jsonl", "w") as f:
            for i in allinfo_dict_list:
                json.dump(i, f)
                f.write("\n")






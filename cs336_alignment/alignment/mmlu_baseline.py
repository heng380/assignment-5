import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import time
import tqdm
import os
import argparse

MODEL_PATH = "/cfs/hengcao/meta-llama"
MMLU_PATH = "/home/ubuntu/hengcao/assignment-5/data/mmlu/test"
PROMPT = 'Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\n'

def load_mmlu_data():
    path = MMLU_PATH
    subject_files = []
    tasks = {}
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith("test.csv"):
                subject_files.append((file.replace(f"_test.csv", ""), os.path.join(root, file)))
    
    for subject, file_path in subject_files:
        df = pd.read_csv(file_path)
        df.columns = ["question", "A", "B", "C", "D", "answer"]
        tasks[subject] = df
    return tasks


def parse_mmlu_answer(response: str):
    """
    Parse model response for MMLU answer.
    Returns "A", "B", "C", "D" or None.
    """
    if not response or not isinstance(response, str):
        return None, False

    # 统一大小写
    text = response.strip().upper()

    # 常见 pattern: "ANSWER: C", "(B)", "选C", "CHOICE D"
    patterns = [
        r"\b([ABCD])\b",             # 单独的 A/B/C/D
        r"ANSWER[:： ]*([ABCD])",    # "Answer: C"
        r"OPTION[:： ]*([ABCD])",    # "Option D"
        r"CHOICE[:： ]*([ABCD])",    # "Choice B"
        r"选项?([ABCD])",             # "选C" 或 "选项C"
        r"\(([ABCD])\)",             # "(A)"
    ]

    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).upper(), True

    # 没找到合法选项
    return None, False

def evaluate_model(model, tokenizer, task):
    all_results = {}
    all_metrics = {}
    parsing_failures = []
    start_time = time.time()
    total_examples = 0

    for subject, examples in task.items():
        subject_results = []
        correct = 0
        total = 0
        subject_failures = 0
        for _, row in tqdm.tqdm(examples.iterrows(), total=len(examples)):
            question = row["question"]
            choices = {"A":row["A"], "B": row["B"], "C":row["C"], "D":row["D"]}
            ground_truth = row["answer"]
            prompt = format_prompt(subject, question, choices)

            input = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            outputs = model.generate(**input, max_new_tokens=16, do_sample=False, top_p=1)
            input_ids = input["input_ids"][0]
            generate_text = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
            model_answer, parsing_succeeded = parse_mmlu_answer(generate_text)
            print (f"prompt:{prompt}, generate_text:{generate_text}, model_answer:{model_answer}, ground_truth:{ground_truth}")

            if not parsing_succeeded:
                subject_failures += 1
                parsing_failures.append({
                    "subject": subject,
                    "question": question,
                    "choices": choices,
                    "ground_truth": ground_truth,
                    "model_output": generate_text,
                    "prompt": prompt
                })
            if model_answer == ground_truth:
                correct += 1
            total += 1
            total_examples += 1
        acc = correct / total
        metrics = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "parsing_failures": subject_failures,
            "parsing_failure_rate": subject_failures/total
        }

        all_metrics[subject] = metrics

        print (f"subject: {subject}, acc: {acc}, parsing failure: {subject_failures}")
    
    end_time = time.time()
    total_time = end_time-start_time

    print (f"total time: {total_time}, total count: {total_examples}")
    return all_results, all_metrics, parsing_failures


def format_prompt(subject, question, choices):
    prompt = PROMPT.format(subject = subject)
    prompt += f"Question: {question}\n"
    for choice_key, choice_value in choices.items():
        prompt += f"{choice_key}. {choice_value}\n"
    prompt += "Answer:"
    return prompt

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code = True
    )

    task = load_mmlu_data()

    all_results, all_metrics, parsing_failures = evaluate_model(model, tokenizer, task)
    print (f"all_metrics: {all_metrics}")
    print (f"all failures: {parsing_failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--choice")
    args = parser.parse_args()

    if args.choice == "load_data":
        task = load_mmlu_data()
        print (task["high_school_government_and_politics"])
    elif args.choice == "format_prompt":
        prompt = format_prompt(subject="high school math", question="1+1=?", choices={"A":1, "B":2, "C":3, "D":4})
        print (prompt)
    else:
        main()

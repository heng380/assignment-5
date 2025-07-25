import json
import os
import re
from typing import List, Callable, Dict
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")


def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_prompt_with_template(question: str, template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(question=question)


def evaluate_vllm(
    data: List[Dict],
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    # 3. Generate outputs for each example
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    generated_texts = [output.outputs[0].text.strip() for output in outputs]

    # 4. Calculate evaluation metrics and collect examples
    results = []
    counts = {
        "correct": 0,
        "wrong_answer": 0,
        "wrong_format": 0
    }
    correct_examples = []
    format_error_examples = []
    answer_error_examples = []

    for prompt, generated_text, example in zip(prompts, generated_texts, data):
        ground_truth = example["answer"]
        reference_answer = extract_reference_answer(ground_truth)
        print ("!!!!!alert!!!!!")
        print (generated_text, reference_answer)
        metrics = reward_fn(generated_text, reference_answer)

        if metrics["format_reward"] == 1 and metrics["answer_reward"] == 1:
            counts["correct"] += 1
            correct_examples.append({
                "prompt": prompt,
                "response": generated_text,
                "reference_answer": ground_truth,
                "reference_answer_extracted": reference_answer
            })
        elif metrics["format_reward"] == 1 and metrics["answer_reward"] == 0:
            counts["wrong_answer"] += 1
            answer_error_examples.append({
                "prompt": prompt,
                "response": generated_text,
                "reference_answer": ground_truth,
                "reference_answer_extracted": reference_answer
            })
        else:
            counts["wrong_format"] += 1
            format_error_examples.append({
                "prompt": prompt,
                "response": generated_text,
                "reference_answer": ground_truth,
                "reference_answer_extracted": reference_answer
            })

        results.append({
            "prompt": prompt,
            "response": generated_text,
            "reference_answer": ground_truth,
            "reference_answer_extracted": reference_answer,
            "metrics": metrics
        })

    # 5. Save results
    os.makedirs("outputs", exist_ok=True)
    OUTPUT_PATH = os.path.join("outputs", "eval_results.jsonl")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Evaluation results saved to {OUTPUT_PATH}")

    return counts, format_error_examples, answer_error_examples, correct_examples


def main():
    # llm
    llm = LLM(model="/home/aiscuser/repos/assignment5-alignment/data/model/Qwen2.5-Math-1.5B")

    # prompts
    # 1. load data/gsm8k/test.jsonl
    data = load_jsonl("/home/aiscuser/repos/assignment5-alignment/data/gsm8k/test.jsonl")

    # 2. format and use r1_zero prompt cs336_alignment/prompts/r1_zero.prompt
    TEMPLATE_PATH = "/home/aiscuser/repos/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    formatted_prompts = [
        format_prompt_with_template(example["question"], TEMPLATE_PATH) for example in data
    ]

    # sampling params
    sampling_params =  SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    # eval
    counts, format_errors, answer_errors, correct_answers = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        data=data,
        prompts=formatted_prompts,
        eval_sampling_params=sampling_params
    )
    print ("overview:", counts)

    print("\n🔴 Examples of format errors:")
    for ex in format_errors[:10]:
        print("- Prompt:", ex["prompt"])
        print("  Response:", ex["response"])
        print("  Reference Answer:", ex["reference_answer"])
        print("  Extracted reference Answer:", ex["reference_answer_extracted"])
        print()

    print("\n🟡 Examples of format OK but wrong answer:")
    for ex in answer_errors[:10]:
        print("- Prompt:", ex["prompt"])
        print("  Response:", ex["response"])
        print("  Reference Answer:", ex["reference_answer"])
        print("  Extracted reference Answer:", ex["reference_answer_extracted"])
        print()

    print("\n✅ Examples of format errors:")
    for ex in correct_answers[:10]:
        print("- Prompt:", ex["prompt"])
        print("  Response:", ex["response"])
        print("  Reference Answer:", ex["reference_answer"])
        print("  Extracted reference Answer:", ex["reference_answer_extracted"])
        print()

    print("\n📊 Evaluation Summary:")
    print(f"Correct (format + answer): {counts['correct']}")
    print(f"Wrong answer (but correct format): {counts['wrong_answer']}")
    print(f"Wrong format: {counts['wrong_format']}")


if __name__ == "__main__":
    print(os.getcwd())
    main()
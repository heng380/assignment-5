"""
    DPO implementation on anthropic harmless/helpful data.
"""
import os
import gzip
import json
import torch
import argparse
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

HH_PATH = "/home/ubuntu/hengcao/assignment-5/data/hh-rlhf"
OUTPUT_DIR = "./out/dpo"
POLICY_MODEL_PATH  = "/home/ubuntu/hengcao/assignment-5/cs336_alignment/alignment/out/sft/checkpoint-6500"
REFERENCE_MODEL_PATH = "/cfs/hengcao/meta-llama"
max_length = 512
train_batch_size = 64
gradient_accumulation_steps = 16
micro_batch_size = train_batch_size // gradient_accumulation_steps  # 修复：使用整数除法
lr=1e-6
epochs=1
beta=0.1
eval_interval = 100
save_interval = 500
policy_device = "cuda:0"
reference_device = "cuda:1"

def load_hh_dataset():
    filenames = [
        "harmless-base.jsonl.gz",
        "helpful-base.jsonl.gz",
        "helpful-online.jsonl.gz",
        "helpful-rejection-sampled.jsonl.gz"
    ]

    all_examples = []

    for filename in filenames:
        file_path = os.path.join(HH_PATH, filename)
        with gzip.open(file_path, "rt") as f:
            for line in f:
                data = json.loads(line)

                chosen_conversation = data.get("chosen", "")
                rejected_conversation = data.get("rejected", "")

                chosen_messages = [msg for msg in chosen_conversation.split("\n\n") if msg.strip()]
                rejected_messages = [msg for msg in rejected_conversation.split("\n\n") if msg.strip()]
                if len(chosen_messages) < 2 or len(rejected_messages) < 2:
                    continue
                if len(chosen_messages) > 2 or len(rejected_messages) > 2:
                    continue

                human_msg_chosen = chosen_messages[0]
                human_msg_reject = rejected_messages[0]

                if human_msg_chosen != human_msg_reject:
                    continue

                chosen_response = chosen_messages[1]
                reject_response = rejected_messages[1]

                if not chosen_response or not reject_response:
                    continue

                example = {
                    "instruction": human_msg_chosen,
                    "chosen": chosen_response,
                    "rejected": reject_response
                }
                all_examples.append(example)
                # print (example)
    random.shuffle(all_examples)
    split_index = int(len(all_examples) * 0.95)
    train_examples = all_examples[:split_index]
    val_examples = all_examples[split_index:]
    return train_examples, val_examples

class HHPreferenceDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]

        instruction = example["instruction"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
        prompt = prompt_template.format(instruction)

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        chosen_tokens = self.tokenizer.encode(chosen+self.tokenizer.eos_token, add_special_tokens=False)
        rejected_tokens = self.tokenizer.encode(rejected + self.tokenizer.eos_token, add_special_tokens=False)

        if len(prompt_tokens) + max(len(chosen_tokens), len(rejected_tokens)) > self.max_length:
            max_prompt_length = self.max_length - max(len(chosen_tokens), len(rejected_tokens))
            prompt_tokens = prompt_tokens[:max_prompt_length]

        chosen_input_ids = prompt_tokens + chosen_tokens
        rejected_input_ids = prompt_tokens + rejected_tokens

        # 保存原始长度用于注意力掩码计算
        chosen_original_length = len(chosen_input_ids)
        rejected_original_length = len(rejected_input_ids)

        # 填充或截断到最大长度
        if len(chosen_input_ids) < self.max_length:
            chosen_input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(chosen_input_ids))
        else:
            chosen_input_ids = chosen_input_ids[:self.max_length]
            chosen_original_length = self.max_length

        if len(rejected_input_ids) < self.max_length:
            rejected_input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(rejected_input_ids))
        else:
            rejected_input_ids = rejected_input_ids[:self.max_length]
            rejected_original_length = self.max_length

        # 修复注意力掩码计算：基于实际有效长度
        chosen_attention_mask = [1] * chosen_original_length + [0] * (self.max_length - chosen_original_length)
        rejected_attention_mask = [1] * rejected_original_length + [0] * (self.max_length - rejected_original_length)

        chosen_input_ids = torch.tensor(chosen_input_ids)
        rejected_input_ids = torch.tensor(rejected_input_ids)
        chosen_attention_mask = torch.tensor(chosen_attention_mask)
        rejected_attention_mask = torch.tensor(rejected_attention_mask)
        
        return {
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_attention_mask": rejected_attention_mask,
            "prompt_length": len(prompt_tokens),
        }



def compute_response_log_probs(logits, input_ids, attention_mask, prompt_length):
    """
    Compute log probabilities for a response, excluding the prompt.
    
    Args:
        logits: Model logits
        input_ids: Input token IDs
        attention_mask: Attention mask
        prompt_length: Length of the prompt to exclude
    
    Returns:
        log_probs: Sum of log probabilities for the response
    """
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Shift for next-token prediction
    shift_log_probs = log_probs[:, :-1, :]
    shift_input_ids = input_ids[:, 1:]
    shift_attention_mask = attention_mask[:, 1:]
    
    # Get the token log probs
    token_log_probs = torch.gather(
        shift_log_probs, 
        dim=2, 
        index=shift_input_ids.unsqueeze(-1)
    ).squeeze(-1)
    
    # Apply attention mask and prompt mask
    batch_size = token_log_probs.shape[0]
    seq_length = token_log_probs.shape[1]
    
    response_mask = torch.zeros_like(shift_attention_mask)
    for i in range(batch_size):
        if isinstance(prompt_length, int):
            response_mask[i, prompt_length-1:] = 1
        else:
            response_mask[i, prompt_length[i]-1:] = 1
    
    # Combine with attention mask
    combined_mask = shift_attention_mask * response_mask
    
    # Mask and sum log probs
    masked_log_probs = token_log_probs * combined_mask
    
    # 检查是否有有效的响应token
    valid_tokens = combined_mask.sum(dim=1)
    
    # 如果没有有效token，返回一个很小的负数（表示极低概率）
    # 这比使用clamp(min=1)更合理，因为它保持了数学意义
    result = torch.where(
        valid_tokens > 0,
        masked_log_probs.sum(dim=1),  # 对于DPO，应该使用总和而不是平均值
        torch.full_like(valid_tokens, -1e8, dtype=torch.float)  # 极小的log概率
    )
    
    return result

def compute_model_logits(policy_model, reference_model, input_ids, attention_mask, policy_device, reference_device):
    """
    通用函数：计算策略模型和参考模型的logits
    
    Args:
        policy_model: 策略模型
        reference_model: 参考模型
        input_ids: 输入token ids
        attention_mask: 注意力掩码
        policy_device: 策略模型设备
        reference_device: 参考模型设备
    
    Returns:
        tuple: (policy_logits, reference_logits)
    """
    # 策略模型前向传播
    policy_logits = policy_model(
        input_ids=input_ids,
        attention_mask=attention_mask
    ).logits
    
    # 参考模型前向传播（处理设备转换）
    ref_input_ids = input_ids.to(reference_device) if policy_device != reference_device else input_ids
    ref_attention_mask = attention_mask.to(reference_device) if policy_device != reference_device else attention_mask
    
    reference_logits = reference_model(
        input_ids=ref_input_ids,
        attention_mask=ref_attention_mask
    ).logits
    
    # 将参考模型输出转回策略模型设备
    if policy_device != reference_device:
        reference_logits = reference_logits.to(policy_device)
    
    return policy_logits, reference_logits

def compute_dpo_rewards_and_loss(chosen_logits, chosen_logits_ref, rejected_logits, rejected_logits_ref,
                                chosen_input_ids, rejected_input_ids, chosen_attention_mask, 
                                rejected_attention_mask, prompt_length, beta):
    """
    通用函数：计算DPO奖励和损失
    
    Args:
        chosen_logits: 选择响应的策略模型logits
        chosen_logits_ref: 选择响应的参考模型logits
        rejected_logits: 拒绝响应的策略模型logits
        rejected_logits_ref: 拒绝响应的参考模型logits
        chosen_input_ids: 选择响应的输入ids
        rejected_input_ids: 拒绝响应的输入ids
        chosen_attention_mask: 选择响应的注意力掩码
        rejected_attention_mask: 拒绝响应的注意力掩码
        prompt_length: 提示长度
        beta: DPO参数
    
    Returns:
        tuple: (loss, chosen_rewards, rejected_rewards)
    """
    # 计算日志概率
    chosen_log_probs = compute_response_log_probs(chosen_logits, chosen_input_ids, chosen_attention_mask, prompt_length)
    chosen_log_probs_ref = compute_response_log_probs(chosen_logits_ref, chosen_input_ids, chosen_attention_mask, prompt_length)
    
    rejected_log_probs = compute_response_log_probs(rejected_logits, rejected_input_ids, rejected_attention_mask, prompt_length)
    rejected_log_probs_ref = compute_response_log_probs(rejected_logits_ref, rejected_input_ids, rejected_attention_mask, prompt_length)
    
    # 计算奖励
    chosen_rewards = chosen_log_probs - chosen_log_probs_ref
    rejected_rewards = rejected_log_probs - rejected_log_probs_ref
    
    # 计算损失
    loss = -F.logsigmoid(beta * (chosen_rewards - rejected_rewards))
    
    return loss, chosen_rewards, rejected_rewards

def train_dpo(
        policy_model_path=POLICY_MODEL_PATH,
        reference_model_path=REFERENCE_MODEL_PATH,
        hh_dir=HH_PATH,
        output_dir=OUTPUT_DIR,
        batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ):
    """
    DPO训练主函数，使用全局配置参数
    """
    # 初始化wandb
    wandb.init(
        project="sft",
        name=f"dpo-experiment-{epochs}epochs-{lr}lr-{beta}beta",
        config={
        }
    )
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
    policy_model = AutoModelForCausalLM.from_pretrained(policy_model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    policy_model.to(policy_device)

    reference_model = AutoModelForCausalLM.from_pretrained(reference_model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    reference_model.to(reference_device)

    reference_model.eval()
    train_examples, val_examples = load_hh_dataset()

    train_dataset = HHPreferenceDataset(train_examples, tokenizer, max_length)
    val_dataset = HHPreferenceDataset(val_examples, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=lr)

    policy_model.train()
    step = 0
    total_loss = 0
    
    # 计算总步数
    total_steps = len(train_dataloader) * epochs / gradient_accumulation_steps
    print(f"start dpo training! 总共 {total_steps} 个step")

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            chosen_input_ids = batch["chosen_input_ids"].to(policy_device)
            rejected_input_ids = batch["rejected_input_ids"].to(policy_device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(policy_device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(policy_device)

            prompt_length = batch["prompt_length"]

            # 使用通用函数计算logits
            chosen_logits, chosen_logits_ref = compute_model_logits(
                policy_model, reference_model, chosen_input_ids, chosen_attention_mask, 
                policy_device, reference_device
            )
            
            rejected_logits, rejected_logits_ref = compute_model_logits(
                policy_model, reference_model, rejected_input_ids, rejected_attention_mask,
                policy_device, reference_device
            )
            
            # 使用通用函数计算奖励和损失
            losses, chosen_rewards, rejected_rewards = compute_dpo_rewards_and_loss(
                chosen_logits, chosen_logits_ref, rejected_logits, rejected_logits_ref,
                chosen_input_ids, rejected_input_ids, chosen_attention_mask, 
                rejected_attention_mask, prompt_length, beta
            )
            
            loss = losses.mean()

            loss /= gradient_accumulation_steps

            loss.backward()

            total_loss = loss.item() * gradient_accumulation_steps

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                step+= 1
                print(f"step {step} loss: {total_loss:.4f}")
                
                # 记录训练指标到wandb
                wandb.log({
                    "train/loss": total_loss
                }, step=step)

                if step % eval_interval == 1:
                    val_loss, val_accuracy = evaluate_dpo(policy_model, reference_model, val_dataloader, beta, policy_device, reference_device)
                    print (f"validate loss: {val_loss:.4f}, acc: {val_accuracy:.4f}")
                    
                    # 记录验证指标到wandb
                    wandb.log({
                        "val/loss": val_loss,
                        "val/accuracy": val_accuracy
                    }, step=step)

                if step % save_interval == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    policy_model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    print (f"Saved checkpoint to {checkpoint_path}")
    
    # 训练结束，完成wandb记录
    wandb.finish()
    print("DPO训练完成！")




def evaluate_dpo(
    policy_model, 
    reference_model, 
    dataloader, 
    beta, 
    policy_device, 
    reference_device
):
    """
    Evaluate a model with DPO on a validation set.
    
    Args:
        policy_model: The policy model
        reference_model: The reference model
        dataloader: The validation dataloader
        beta: The regularization strength
        policy_device: Device for the policy model
        reference_device: Device for the reference model
    
    Returns:
        avg_loss: The average validation loss
        accuracy: The classification accuracy
    """
    policy_model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to correct device
            chosen_input_ids = batch["chosen_input_ids"].to(policy_device)
            rejected_input_ids = batch["rejected_input_ids"].to(policy_device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(policy_device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(policy_device)
            prompt_length = batch["prompt_length"]
            
            # 使用通用函数计算logits
            chosen_logits, chosen_logits_ref = compute_model_logits(
                policy_model, reference_model, chosen_input_ids, chosen_attention_mask,
                policy_device, reference_device
            )
            
            rejected_logits, rejected_logits_ref = compute_model_logits(
                policy_model, reference_model, rejected_input_ids, rejected_attention_mask,
                policy_device, reference_device
            )
            
            # 使用通用函数计算奖励和损失
            losses, chosen_rewards, rejected_rewards = compute_dpo_rewards_and_loss(
                chosen_logits, chosen_logits_ref, rejected_logits, rejected_logits_ref,
                chosen_input_ids, rejected_input_ids, chosen_attention_mask,
                rejected_attention_mask, prompt_length, beta
            )
            
            # Compute accuracy (do we prefer the chosen response?)
            correct = (chosen_rewards > rejected_rewards).float()
            
            # Update stats
            total_loss += losses.sum().item()
            total_correct += correct.sum().item()
            total_examples += chosen_input_ids.size(0)
    
    # Reset model to training mode
    policy_model.train()
    
    # Compute average loss and accuracy
    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    
    return avg_loss, accuracy




def main():
    train_dpo()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--choice")
    args = parser.parse_args()

    if args.choice == "load_data":
        task = load_hh_dataset()
        print (task["high_school_government_and_politics"])
    # elif args.choice == "format_prompt":
    #     prompt = format_prompt(subject="high school math", question="1+1=?", choices={"A":1, "B":2, "C":3, "D":4})
    #     print (prompt)
    else:
        main()

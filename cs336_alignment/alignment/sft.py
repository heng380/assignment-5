"""
    sft on ultrachat and llama safety prompt-response pair data
"""
import math
from torch.optim.lr_scheduler import LambdaLR
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from instruction_finetune import get_sft_dataloader, get_packed_sft_dataloader # 确保这个导入路径正确
from torch.optim import AdamW
import torch.nn.functional as F
import wandb
import time # 导入 time 用于计时

# 设置环境变量以避免 tokenizers 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/cfs/hengcao/meta-llama"
TRAIN_PATH = "/home/ubuntu/hengcao/assignment-5/data/train.jsonl"
TEST_PATH = "/home/ubuntu/hengcao/assignment-5/data/test.jsonl"
OUT_DIR = "./out/sft"
seq_length = 512
train_batch_size = 32
gradient_accumulation_steps = 8
micro_batch_size = int(train_batch_size / gradient_accumulation_steps)
epochs = 1
lr = 2e-5
warmup_ratio = 0.03
log_interval = 1
eval_interval = 100
save_interval = 500
device = "cuda"
# --- 新增：用于控制评估进度打印的间隔 ---
eval_log_interval = 10 # 每处理 10 个 batch 打印一次进度
# --- ---


def get_linear_warmup_cosine_decay_scheduler(optimizer, warmup_step, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_step:
            return float(current_step) / float(max(1, warmup_step))
        progress = float(current_step - warmup_step) / float(max(1, total_steps - warmup_step))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def compute_entropy(logits, labels):
    # logits: [batch, seq_len, vocab]
    # labels: [batch, seq_len]
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]
    # 只统计labels不为-100的token
    mask = (labels != -100)
    if mask.sum() == 0:
        return 0.0
    masked_entropy = entropy[mask]
    return masked_entropy.mean().item()

def evaluate(model, dataloader, tokenizer, device, step=None, log_wandb=False, prefix="val"):
    print(f"Starting evaluation at step {step}...") # --- 新增：开始评估提示 ---
    start_time = time.time() # --- 新增：记录开始时间 ---
    model.eval()
    total_loss = 0
    total_examples = 0
    total_entropy = 0
    num_batches = 0

    # --- 新增：获取总 batch 数用于进度显示 ---
    total_val_batches = len(dataloader)
    print(f"Total validation batches: {total_val_batches}") # --- 新增：打印总 batch 数 ---
    # --- ---


    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # --- 新增：评估进度打印 ---
            if (batch_idx + 1) % eval_log_interval == 0 or (batch_idx + 1) == total_val_batches:
                 print(f"Evaluating: processed {batch_idx + 1}/{total_val_batches} batches...")
            if (batch_idx + 1) == 30:
                break 
            # --- ---

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits

            # --- 优化：直接使用 logits 和 labels ---
            # shift_logits = logits
            # shift_labels = labels

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # 使用 logits
                labels.view(-1),                  # 使用 labels
                ignore_index=-100
            )

            # 累积 loss 和 example 数量
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            # 计算并累积 entropy
            entropy = compute_entropy(logits, labels) # 使用 logits 和 labels
            total_entropy += entropy * batch_size # 加权平均
            num_batches += 1

    model.train()
    # --- 优化：使用加权平均计算 avg_loss 和 avg_entropy ---
    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
    avg_entropy = total_entropy / total_examples if total_examples > 0 else 0.0
    # --- ---
    
    # --- 新增：结束评估提示和耗时 ---
    end_time = time.time()
    eval_duration = end_time - start_time
    print(f"Evaluation completed in {eval_duration:.2f} seconds.")
    # --- ---

    if log_wandb and step is not None:
        wandb.log({
            f"{prefix}/loss": avg_loss,
        }, step=step)
        wandb.log({
            f"{prefix}/entropy": avg_entropy,
        }, step=step)

    return avg_loss, avg_entropy

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wandb.init(project="sft", name="sft_run")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.to(device)
    print ("model loaded!")

    # 注意：这里加载训练和验证数据集都用了 TEST_PATH，这可能是笔误？应该是 TRAIN_PATH 吗？
    # train_dataset = get_packed_sft_dataloader(tokenizer, TEST_PATH, seq_length, shuffle=True) 
    # val_dataset = get_packed_sft_dataloader(tokenizer, TEST_PATH, seq_length, shuffle=False)
    # --- 修正：使用正确的路径 ---
    train_dataset = get_packed_sft_dataloader(tokenizer, TRAIN_PATH, seq_length, shuffle=True)
    val_dataset = get_packed_sft_dataloader(tokenizer, TEST_PATH, seq_length, shuffle=False)
    # --- ---

    train_dataloader = get_sft_dataloader(
        train_dataset, 
        micro_batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_dataloader = get_sft_dataloader(
        val_dataset, 
        25, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    print ("data loaded")
    optimizer = AdamW(model.parameters(), lr = lr)

    total_steps = epochs * len(train_dataloader) // gradient_accumulation_steps
    print (f"total step: {total_steps}")
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_warmup_cosine_decay_scheduler(
        optimizer, warmup_steps, total_steps
    )

    model.train()
    step = 0
    total_loss = 0

    print ("start training!")
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids = input_ids)
            logits = outputs.logits

            # shift_logits = logits
            # shift_labels = labels

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # 使用 logits
                labels.view(-1),                  # 使用 labels
                ignore_index=-100
            )

            loss/=gradient_accumulation_steps

            loss.backward()

            total_loss += loss.item()

            train_entropy = compute_entropy(logits, labels) # 使用 logits 和 labels

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step += 1

                if step % log_interval == 0:
                    avg_loss = total_loss / log_interval
                    total_loss = 0
                    print (f"step {step}, training loss: {avg_loss}")
                    wandb.log({
                        "train/loss": avg_loss,
                    }, step=step)
                    wandb.log({
                        "train/entropy": train_entropy,
                    }, step=step)

                if step % eval_interval == 0:
                    # --- 修改：添加评估开始/结束提示 ---
                    print(f"Triggering evaluation at step {step}...")
                    val_loss, val_entropy = evaluate(model, val_dataloader, tokenizer, device, step=step, log_wandb=True, prefix="val")
                    print (f"step {step}, eval loss: {val_loss}, eval entropy: {val_entropy}")
                    print(f"Evaluation at step {step} finished.")

                if step % save_interval == 0:
                    checkpoint_path = os.path.join(OUT_DIR, f"checkpoint-{step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    print ("saved model!")

if __name__ == "__main__":
    main()




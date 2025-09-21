import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import os

# load alpaca prompt
with open('/home/ubuntu/hengcao/assignment-5/cs336_alignment/prompts/alpaca_sft.prompt', 'r') as f:
    ALPACA_PROMPT = f.read()
class PackedSFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.dataset_path = dataset_path
        
        # 读取数据集
        with open(dataset_path, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f]
        
        # 如果需要，打乱文档顺序
        if shuffle:
            random.shuffle(examples)
        
        # 使用Alpaca模板格式化每个(prompt, response)对
        template = ALPACA_PROMPT
        
        # 获取特殊token ID
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id
        
        # 构建所有token序列，将所有文档连接成一个单一序列
        self.all_token_ids = []
        self.prompt_token_lens = []  # 新增：记录每个example的prompt token长度
        for i, example in enumerate(examples):
            # 格式化文本
            formatted_text = template.format(instruction=example["prompt"], response=example["response"])
            # prompt部分格式化
            prompt_text = template.format(instruction=example["prompt"], response="")
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            self.prompt_token_lens.append(len(prompt_tokens))
            
            tokens = tokenizer(formatted_text, add_special_tokens=False)["input_ids"]  # 修复：补回tokens定义
            # 如果是第一个example，在开头添加BOS token
            if i == 0 and bos_token_id is not None:
                self.all_token_ids.append(bos_token_id)
            
            # 添加当前example的tokens
            self.all_token_ids.extend(tokens)
            
            # 在每个example后添加EOS token作为分隔符
            if eos_token_id is not None:
                self.all_token_ids.append(eos_token_id)
                
            # 为下一个example添加BOS token（除了最后一个）
            if i < len(examples) - 1 and bos_token_id is not None:
                self.all_token_ids.append(bos_token_id)
        
        # 将长序列分割成固定长度的块（non-overlapping chunks）
        self.sequences = []
        total_tokens = len(self.all_token_ids)
        
        # 使用非重叠的方式分割序列
        i = 0
        while i + seq_length <= total_tokens:
            chunk = self.all_token_ids[i:i + seq_length]
            self.sequences.append(chunk)
            i += seq_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence, dtype=torch.long)
        
        # 对于语言建模，labels是input_ids向左移位一位
        # input_ids: [token1, token2, token3, ..., tokenN]
        # labels:    [token2, token3, token4, ..., tokenN+1]
        
        # labels的最后一个位置应该是下一个token
        if idx < len(self.sequences) - 1:
            next_sequence = self.sequences[idx + 1]
            labels = torch.tensor(sequence[1:] + [next_sequence[0]], dtype=torch.long)
        else:
            # 对于最后一个序列，从原始token序列中获取下一个token
            start_pos = idx * self.seq_length
            end_pos = start_pos + self.seq_length
            
            # 获取下一个token
            if end_pos < len(self.all_token_ids):
                next_token = self.all_token_ids[end_pos]
                labels = torch.tensor(sequence[1:] + [next_token], dtype=torch.long)
            else:
                # 如果真的没有下一个token，使用-100
                labels = torch.tensor(sequence[1:] + [-100], dtype=torch.long)
        
        # 计算当前sequence对应的example编号和在example内的起始位置
        # 由于all_token_ids是拼接的，需找到当前sequence属于哪个example
        # 简化：假设每个example不会跨多个sequence（如需更精确可进一步追踪）
        # 这里采用近似做法：每seq_length为一个example
        example_idx = idx
        prompt_len = self.prompt_token_lens[example_idx] if example_idx < len(self.prompt_token_lens) else 0

        # 现在labels和input_ids等长
        labels = labels.clone()
        # mask prompt部分
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "labels": labels
        }
    
def get_sft_dataloader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler
    )

def get_packed_sft_dataloader(tokenizer, dataset_path, seq_length, shuffle):
    return PackedSFTDataset(tokenizer, dataset_path, seq_length, shuffle)

def run_iterate_batches(dataset, batch_size, shuffle=True):
    dataloader = get_sft_dataloader(dataset, batch_size, shuffle)
    return list(dataloader)
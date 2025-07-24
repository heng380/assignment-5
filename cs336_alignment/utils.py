import torch
from transformers import PreTrainedTokenizerBase

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase) -> dict[str, torch.Tensor]:
    prompt_input_ids = []
    output_input_ids = []

    for prompt in prompt_strs:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_input_ids.append(torch.tensor(tokens))
    for output in output_strs:
        tokens = tokenizer.encode(output, add_special_tokens=False)
        output_input_ids.append(torch.tensor(tokens))
    
    seq_lengths = [len(p_ids) + len(o_ids) for p_ids, o_ids in zip(prompt_input_ids, output_input_ids)]
    max_length = max(seq_lengths)

    concatenated_input_ids = []
    concatenated_labels = []
    response_masks = []

    for p_ids, o_ids, in zip(prompt_input_ids, output_input_ids):
        input_ids = torch.cat([p_ids, o_ids], dim=0)
        response_mask = torch.cat([
            torch.zeros_like(p_ids, dtype=torch.bool),
            torch.ones_like(o_ids, dtype=torch.bool)
        ], dim=0)
        pad_length = max_length - input_ids.shape[0]
        padded_input_ids = torch.nn.functional.pad(input_ids, (0, pad_length), value=tokenizer.pad_token_id)
        padded_response_mask = torch.nn.functional.pad(response_mask, (0, pad_length), value=False)

        concatenated_input_ids.append(padded_input_ids[:-1])
        concatenated_labels.append(padded_input_ids[1:])
        response_masks.append(padded_response_mask[1:])    ### set prompt and padding to be false, will not calculate loss

    input_ids_tensor = torch.stack(concatenated_input_ids)
    label_tensor = torch.stack(concatenated_labels)
    response_mask_tensor = torch.stack(response_masks)

    return {
        "input_ids": input_ids_tensor,
        "labels": label_tensor,
        "response_mask": response_mask_tensor
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    ### logits: b s v
    ### return: b s
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    p = torch.exp(logp)
    ce = -torch.sum(p*logp, dim=-1)
    return ce
    

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

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits   # b s v
    log_softmax = torch.nn.functional.log_softmax(logits)  # b s v
    label_token_log_softmax = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    if return_token_entropy:
        entropy = compute_entropy(logits)
        return {
            "log_probs": label_token_log_softmax,
            "token_entropy": entropy
        }
    else:
        return {
            "log_probs": label_token_log_softmax
        }

def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None=None,
        normalize_constant: float = 1.0,
) -> torch.Tensor:
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor)) # b s v
    return torch.sum(masked_tensor, dim=dim) / normalize_constant

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    ### logits: b s v
    ### return: b s
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    p = torch.exp(logp)
    ce = -torch.sum(p*logp, dim=-1)
    return ce
    
def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    masked_normalized_probs = masked_normalize(
        policy_log_probs, response_mask, -1, normalize_constant
    )
    loss = -masked_normalized_probs.mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss, {}

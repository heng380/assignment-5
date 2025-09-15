import torch
from einops import repeat
from typing import Literal
from utils import masked_normalize
def compute_group_normalized_reward(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalized_by_std
):
    raw_rewards = []
    for rollout_response, gt_response in zip(rollout_responses, repeated_ground_truths):
        curr_reward = reward_fn(rollout_response, gt_response)["reward"]
        raw_rewards.append(curr_reward)

    raw_rewards = torch.tensor(raw_rewards)  # prompts * group_size, 1
    rewards_per_group = raw_rewards.reshape((-1, group_size))     # prompts * group_size
    mean_reward_per_group = torch.mean(rewards_per_group, dim=-1, keepdim=True)   # prompts * 1
    advantage = rewards_per_group - mean_reward_per_group   # prompts * group_size

    if normalized_by_std:
        std_reward_per_group = torch.std(rewards_per_group, dim=-1, keepdim=True)
        advantage /= (std_reward_per_group + advantage_eps)   # prompts
    advantage = advantage.flatten()   # prompts * group_size, 1

    metadata = {
        'mean': torch.mean(raw_rewards),
        'std': torch.std(raw_rewards),
        'max': torch.max(raw_rewards),
        'min': torch.min(raw_rewards)
    }

    return advantage, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,    # batch_size, 1
    policy_log_probs: torch.Tensor       # batch_size,  seq_len
) -> torch.Tensor: 
    batch_size, seq_len = policy_log_probs.shape
    raw_rewards_or_advantages = repeat(raw_rewards_or_advantages, 'b 1->b s', s=seq_len)
    loss = -raw_rewards_or_advantages * policy_log_probs

    return loss

def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    batch_size, seq_len = policy_log_probs.shape
    advantages = repeat(advantages, "b 1 -> b s", s=seq_len)
    v = pi_ratio * advantages

    meta = {"cliped": v>=10}
    return -v, meta
 
def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    batch_size, seq_len = policy_log_probs.shape
    advantages = repeat(advantages, "b 1 -> b s", s=seq_len)
    v = pi_ratio * advantages
    v_clip = torch.clip(pi_ratio, min=1-cliprange, max=1+cliprange) * advantages

    meta = {
        "cliped": v > v_clip
    }
    return -torch.min(v, v_clip), meta

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None=None,
    advantages: torch.Tensor | None=None,
    old_log_probs: torch.Tensor | None=None,
    cliprange: float | None=None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    elif loss_type == "grpo_no_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_no_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    elif loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None
) -> torch.Tensor:
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    return torch.sum(masked_tensor, dim=dim) / torch.sum(mask, dim=dim)    # 进行了长度归一化, 为per token的loss


def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None=None,
        advantages: torch.Tensor | None=None,
        old_log_probs: torch.Tensor | None=None,
        cliprange: float | None=None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(loss, response_mask)
    loss /= gradient_accumulation_steps  # loss是相加的, 所以需要除以steps
    loss.backward()
    return loss, metadata

def grpo_microbatch_train_step_seq_level_loss(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None=None,
        advantages: torch.Tensor | None=None,
        old_log_probs: torch.Tensor | None=None,
        cliprange: float | None=None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    constant = response_mask.sum(dim=-1).max().item()
    loss = masked_normalize(loss, response_mask, dim=-1, normalize_constant=constant).mean()
    loss /= gradient_accumulation_steps  # loss是相加的, 所以需要除以steps
    loss.backward()
    return loss, metadata

def gradient_clipping(model):
    params_gradients = []
    for param in model.parameters():
        if param.grad is not None:
            params_gradients.append(param.grad.data.flatten())
    grads = torch.cat(params_gradients)
    if torch.norm(grads) > 1.0:
        norm = torch.norm(grads)
        for param in model.parameters():
            param.grad.data /= norm

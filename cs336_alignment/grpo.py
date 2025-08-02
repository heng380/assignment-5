import torch
from einops import repeat
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
    policy_log_probs: torch.Tensor       # batch_size * seq_len
) -> torch.Tensor: 
    batch_size, seq_len = policy_log_probs.shape
    raw_rewards_or_advantages = repeat(raw_rewards_or_advantages, 'b 1->b s', s=seq_len)
    loss = -raw_rewards_or_advantages * policy_log_probs

    return loss

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

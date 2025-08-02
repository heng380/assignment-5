import torch

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
        advantage /= (std_reward_per_group + advantage_eps)   # prompts * group_size
    advantage = advantage.flatten()

    metadata = {
        'mean': torch.mean(raw_rewards),
        'std': torch.std(raw_rewards),
        'max': torch.max(raw_rewards),
        'min': torch.min(raw_rewards)
    }

    return advantage, raw_rewards, metadata


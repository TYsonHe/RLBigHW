# A2C/algorithms/rollout_buffer.py
"""A2C 经验回放缓冲区，支持 GAE 优势估计。"""
import numpy as np
import torch
from typing import Tuple


class RolloutBuffer:
    """存储一条 rollout 的转移，并计算 GAE 优势与回报。"""

    def __init__(self):
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.values: list = []
        self.log_probs: list = []
        self.dones: list = []

    def add(self, state: np.ndarray, action: int, reward: float,
            value: float, log_prob: float, done: float):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, next_value: float, gamma: float,
                    gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """使用 GAE 计算优势函数和回报。

        Args:
            next_value: 最后一个状态的 V(s_{T+1})，若终止则为 0.0
            gamma: 折扣因子
            gae_lambda: GAE 平滑参数

        Returns:
            (advantages, returns): 优势数组与回报数组
        """
        advantages = []
        gae = 0.0
        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t]
                     + gamma * values[t + 1] * (1.0 - self.dones[t])
                     - values[t])
            gae = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def get_tensors(self, device: str) -> Tuple[torch.Tensor, ...]:
        """将缓冲区数据转换为 PyTorch 张量。"""
        return (
            torch.FloatTensor(np.array(self.states)).to(device),
            torch.LongTensor(self.actions).to(device),
            torch.FloatTensor(np.array(self.log_probs)).to(device),
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)

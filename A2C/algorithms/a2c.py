# A2C/algorithms/a2c.py
"""Advantage Actor-Critic (A2C) 算法实现。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict

from .rollout_buffer import RolloutBuffer


class ActorCritic(nn.Module):
    """共享骨干的 Actor-Critic 网络。

    结构：共享骨干 → 策略头 (36 类 logits) + 价值头 (1 标量)
    """

    def __init__(self, state_dim: int = 14, action_dim: int = 36,
                 hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        # 共享骨干
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
            ])
            prev_dim = h
        self.backbone = nn.Sequential(*layers)

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value


class A2CAgent:
    """A2C Agent：策略梯度 + 价值函数基线 + GAE 优势估计。"""

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 36,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 7e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.network = ActorCritic(state_dim, action_dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        self._episode_policy_loss = []
        self._episode_value_loss = []
        self._episode_entropy = []

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> Dict:
        """选择动作并返回相关信息。

        Returns:
            dict: {"action": int, "log_prob": float, "value": float}
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.network(state_t)

        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=1).item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
        else:
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        return {
            "action": action,
            "log_prob": log_prob,
            "value": value.item(),
        }

    def learn(self, next_value: float) -> Dict[str, float]:
        """基于当前缓冲区执行一次 A2C 更新。

        Args:
            next_value: V(s_{T+1})，终止时传 0.0

        Returns:
            各项损失的字典
        """
        advantages, returns = self.buffer.compute_gae(
            next_value, self.gamma, self.gae_lambda
        )

        states_t, actions_t, _ = self.buffer.get_tensors(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # 标准化优势
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # 前向传播
        logits, values = self.network(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy()

        # 策略损失：最大化 log π * A → 最小化 -log π * A
        policy_loss = -(log_probs * advantages_t).mean()

        # 价值损失
        value_loss = F.mse_loss(values.squeeze(-1), returns_t)

        # 熵奖励（最大化熵 → 最小化 -熵）
        entropy_loss = -entropy.mean()

        # 总损失
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # 记录本次更新损失
        self._episode_policy_loss.append(policy_loss.item())
        self._episode_value_loss.append(value_loss.item())
        self._episode_entropy.append(entropy.mean().item())

        self.buffer.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": loss.item(),
        }

    def get_episode_loss_avg(self) -> Dict[str, Optional[float]]:
        """获取当前 episode 内所有更新的平均损失，并清空记录。"""
        result = {}
        for name, lst in [("policy_loss", self._episode_policy_loss),
                          ("value_loss", self._episode_value_loss),
                          ("entropy", self._episode_entropy)]:
            result[name] = float(np.mean(lst)) if lst else None
            lst.clear()
        return result

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

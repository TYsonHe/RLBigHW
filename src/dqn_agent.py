import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    """DQN神经网络架构"""

    def __init__(self, state_size, action_size):
        """
        初始化DQN网络
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """前向传播"""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """DQN智能体"""

    def __init__(self, state_size, action_size):
        """
        初始化DQN智能体
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # 经验回放缓冲区
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.HuberLoss()  # Huber损失函数

    def _build_model(self):
        """构建神经网络模型"""
        model = DQN(self.state_size, self.action_size)
        return model

    def update_target_model(self):
        """更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """选择动作（ε-贪婪策略）"""
        if np.random.rand() <= self.epsilon:
            # 随机探索
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        """经验回放训练"""
        if len(self.memory) < batch_size:
            return

        # 从回放缓冲区采样
        minibatch = random.sample(self.memory, batch_size)

        # 准备训练数据
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算损失
        loss = self.loss_fn(current_q.squeeze(), target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()

# 训练函数
def train_dqn(env, agent, episodes=1000, batch_size=32, target_update_freq=10):
    """训练DQN智能体"""
    rewards = []
    losses = []

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        total_reward = 0
        done = False

        while not done:
            # 选择动作
            action = agent.act(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])

            # 存储经验
            agent.remember(state, action, reward, next_state, done)

            # 更新状态
            state = next_state
            total_reward += reward

            # 经验回放
            loss = agent.replay(batch_size)
            if loss is not None:
                losses.append(loss)

        # 定期更新目标网络
        if episode % target_update_freq == 0:
            agent.update_target_model()

        rewards.append(total_reward)
        print(f"Episode: {episode+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    return rewards, losses
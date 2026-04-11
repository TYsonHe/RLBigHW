import gym
from gym import spaces
import numpy as np
import pandas as pd

class ResourceAllocationEnv(gym.Env):
    """强化学习环境：异构资源分配优化"""

    def __init__(self, dataset_path, max_steps=1000):
        super(ResourceAllocationEnv, self).__init__()

        # 加载数据集
        self.dataset = pd.read_csv(dataset_path)
        self.max_steps = max_steps
        self.current_step = 0

        # 定义状态空间维度
        # [节点类型, CPU使用率, 内存使用率, GPU使用率, 网络延迟, 请求类型, 输入长度, 输出长度, 模型类型]
        self.state_dim = 9

        # 定义动作空间
        # 动作0: 选择部署节点 (0=边缘, 1=云端)
        # 动作1: 资源分配级别 (0=低, 1=中, 2=高)
        # 动作2: 实例数量调整 (-1=减少, 0=保持, 1=增加)
        self.action_space = spaces.MultiDiscrete([2, 3, 3])

        # 定义观察空间
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([1, 100, 100, 100, 100, 3, 2048, 512, 3]),
            dtype=np.float32
        )

        # 奖励函数权重
        self.weights = {
            'throughput': 0.3,
            'latency': 0.25,
            'utilization': 0.2,
            'cost': 0.15,
            'error_rate': 0.1
        }

    def reset(self):
        """重置环境到初始状态"""
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        """获取下一个观察状态"""
        if self.current_step >= len(self.dataset):
            self.current_step = 0

        row = self.dataset.iloc[self.current_step]

        # 构建状态向量
        state = np.array([
            0 if row['节点类型'] == '边缘' else 1,  # 节点类型
            row['CPU使用率(%)'],                  # CPU使用率
            row['内存使用率(%)'],                 # 内存使用率
            row['GPU使用率(%)'],                  # GPU使用率
            row['网络延迟(ms)'],                 # 网络延迟
            self._map_request_type(row['请求类型']), # 请求类型
            row['输入长度'],                      # 输入长度
            row['输出长度'],                      # 输出长度
            self._map_model_type(row['模型类型'])   # 模型类型
        ])

        return state

    def step(self, action):
        """执行动作并返回结果"""
        self.current_step += 1

        # 检查是否结束
        done = self.current_step >= self.max_steps

        # 获取当前状态
        state = self._next_observation()

        # 计算奖励
        reward = self._calculate_reward(action)

        return state, reward, done, {}

    def _calculate_reward(self, action):
        """计算奖励值"""
        # 这里简化了奖励计算，实际应用中应根据动作和状态计算真实指标
        # 在实际应用中，这些值应从环境中计算得出

        # 模拟指标值
        throughput = np.random.uniform(0.7, 1.0)
        latency = np.random.uniform(0.1, 0.5)
        utilization = np.random.uniform(0.6, 0.9)
        cost = np.random.uniform(0.1, 0.3)
        error_rate = np.random.uniform(0.05, 0.2)

        # 计算加权奖励
        reward = (
            self.weights['throughput'] * throughput +
            self.weights['latency'] * (1 - latency) +
            self.weights['utilization'] * utilization +
            self.weights['cost'] * (1 - cost) +
            self.weights['error_rate'] * (1 - error_rate)
        )

        return reward

    def _map_request_type(self, request_type):
        """将请求类型映射为数字"""
        mapping = {
            "文本生成": 0,
            "问答": 1,
            "摘要生成": 2,
            "代码生成": 3
        }
        return mapping.get(request_type, 0)

    def _map_model_type(self, model_type):
        """将模型类型映射为数字"""
        mapping = {
            "LLaMA-7B": 0,
            "LLaMA-13B": 1,
            "GPT-3": 2,
            "GPT-4": 3
        }
        return mapping.get(model_type, 0)

# 环境使用示例
if __name__ == "__main__":
    # 创建环境
    env = ResourceAllocationEnv("Datasets/train_dataset.csv")

    # 重置环境
    state = env.reset()

    # 运行一个episode
    done = False
    total_reward = 0

    while not done:
        # 随机选择动作
        action = env.action_space.sample()

        # 执行动作
        state, reward, done, _ = env.step(action)
        total_reward += reward

        print(f"动作: {action}, 奖励: {reward:.4f}, 总奖励: {total_reward:.4f}")

    print(f"Episode完成，总奖励: {total_reward:.4f}")
"""
Gymnasium 环境核心骨架：LLM 集群调度环境。
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.envs.reward_fn import calculate_reward, validate_action_safety


class LLMClusterEnv(gym.Env):
    """LLM 集群调度环境。"""

    def __init__(
        self,
        data_pipeline=None,
        num_nodes: int = 100,
        regions: list = None,
        reward_weights: dict = None
    ):
        """初始化环境。

        Args:
            data_pipeline: 数据管道实例
            num_nodes: 节点数量
            regions: 支持的区域列表
            reward_weights: 奖励权重配置
        """
        super(LLMClusterEnv, self).__init__()

        # 默认配置
        if regions is None:
            regions = ["beijing", "shanghai", "guangzhou"]
        if reward_weights is None:
            reward_weights = {
                'throughput': 0.3,
                'latency': 0.3,
                'utilization': 0.2,
                'cost': 0.1,
                'sla': 0.1
            }

        self.num_nodes = num_nodes
        self.regions = regions
        self.reward_weights = reward_weights
        self.data_pipeline = data_pipeline

        # 状态空间：根据项目规范，状态向量维度约为 256
        # [0:64] 节点资源状态
        # [64:128] 工作负载统计特征
        # [128:192] 地理位置编码
        # [192:224] 时间特征编码
        # [224:256] 全局统计
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32
        )

        # 动作空间：离散动作空间，每个节点对应一个动作
        # 根据项目规范，使用简单的节点ID选择
        self.action_space = gym.spaces.Discrete(num_nodes)

        # 初始化内部状态
        self.current_step = 0
        self.node_states = None
        self.workload_queue = None
        self.global_info = None
        self.episode_reward = 0.0

        # 加载初始数据（如果提供了数据管道）
        if self.data_pipeline is not None:
            self._load_initial_data()

    def _load_initial_data(self):
        """加载初始数据。"""
        try:
            # 加载训练集集群拓扑
            cluster_df = self.data_pipeline.load_cluster("train")

            # 如果节点数量不足，重复使用或截断
            if len(cluster_df) < self.num_nodes:
                repeats = (self.num_nodes // len(cluster_df)) + 1
                cluster_df = pd.concat([cluster_df] * repeats, ignore_index=True)

            cluster_df = cluster_df.head(self.num_nodes)

            # 提取节点状态
            self.node_states = {
                'cpu_cores': cluster_df['cpu_cores'].values,
                'mem_gb': cluster_df['mem_gb'].values,
                'gpu_mem_gb': cluster_df['gpu_mem_gb'].values,
                'region': cluster_df['region'].values,
                'cost_per_hour': cluster_df['cost_per_hour'].values,
                'network_bandwidth_mbps': cluster_df['network_bandwidth_mbps'].values
            }

            # 获取全局信息
            self.global_info = self.data_pipeline.get_global_info("train")

        except Exception as e:
            print(f"Warning: Failed to load initial data: {e}")
            # 使用默认值初始化
            self._initialize_default_state()

    def _initialize_default_state(self):
        """使用默认值初始化状态。"""
        self.node_states = {
            'cpu_cores': np.full(self.num_nodes, 64, dtype=np.int32),
            'mem_gb': np.full(self.num_nodes, 256, dtype=np.int32),
            'gpu_mem_gb': np.random.choice([24, 48, 80], self.num_nodes),
            'region': np.random.choice(self.regions, self.num_nodes),
            'cost_per_hour': np.random.uniform(1.0, 3.5, self.num_nodes),
            'network_bandwidth_mbps': np.random.choice([5000, 10000, 20000], self.num_nodes)
        }

        self.global_info = {
            'total_requests': 1000,
            'avg_input_tokens': 1000,
            'avg_output_tokens': 500,
            'avg_sla_deadline': 200,
            'model_distribution': {'LLaMA3-7B': 0.5, 'Qwen-14B': 0.3, 'DeepSeek-70B': 0.2},
            'priority_distribution': {'high': 0.1, 'medium': 0.3, 'low': 0.6}
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置环境到初始状态。

        Returns:
            observation: 初始观测值
            info: 附加信息字典
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_reward = 0.0

        # 重置工作负载队列（模拟当前待处理请求）
        self.workload_queue = {
            'model_type': np.random.choice(['LLaMA3-7B', 'Qwen-14B', 'DeepSeek-70B']),
            'input_tokens': np.random.randint(50, 4096),
            'output_tokens': np.random.randint(50, 2048),
            'priority': np.random.choice(['high', 'medium', 'low'], p=[0.1, 0.3, 0.6]),
            'sla_deadline_ms': np.random.choice([100, 200, 400, 800])
        }

        # 编码观测值
        observation = self._encode_observation()

        info = {
            'node_states': self.node_states,
            'workload_queue': self.workload_queue,
            'global_info': self.global_info
        }

        return observation, info

    def _encode_observation(self) -> np.ndarray:
        """将内部状态编码为固定维度的观测向量。

        根据项目规范实现 ~256 维的状态编码。
        """
        obs = np.zeros(256, dtype=np.float32)

        # [0:64] 节点资源状态（每个节点 8 维 × 最多 8 节点或统计特征）
        # 由于节点数量可能很多，这里使用统计特征而不是每个节点的具体状态
        num_nodes_to_encode = min(8, self.num_nodes)

        for i in range(num_nodes_to_encode):
            start_idx = i * 8
            obs[start_idx] = self.node_states['cpu_cores'][i] / 256.0  # 归一化 CPU 核数
            obs[start_idx + 1] = self.node_states['mem_gb'][i] / 1024.0  # 归一化内存
            obs[start_idx + 2] = self.node_states['gpu_mem_gb'][i] / 80.0  # 归一化 GPU 显存
            obs[start_idx + 3] = self.node_states['cost_per_hour'][i] / 3.5  # 归一化成本
            obs[start_idx + 4] = self.node_states['network_bandwidth_mbps'][i] / 20000.0  # 归一化带宽

            # 区域 one-hot 编码（简化为数值）
            region_idx = self.regions.index(self.node_states['region'][i]) if self.node_states['region'][i] in self.regions else 0
            obs[start_idx + 5] = region_idx / len(self.regions)

            # 资源利用率（模拟值）
            obs[start_idx + 6] = np.random.uniform(0.2, 0.8)
            obs[start_idx + 7] = np.random.uniform(0.1, 0.9)  # 队列长度归一化

        # [64:128] 工作负载统计特征（64 维）
        # 这里简化为关键统计特征的重复填充
        workload_features = [
            self.workload_queue['input_tokens'] / 4096.0,
            self.workload_queue['output_tokens'] / 2048.0,
            1.0 if self.workload_queue['model_type'] == 'LLaMA3-7B' else 0.0,
            1.0 if self.workload_queue['model_type'] == 'Qwen-14B' else 0.0,
            1.0 if self.workload_queue['model_type'] == 'DeepSeek-70B' else 0.0,
            1.0 if self.workload_queue['priority'] == 'high' else 0.0,
            1.0 if self.workload_queue['priority'] == 'medium' else 0.0,
            self.workload_queue['sla_deadline_ms'] / 1500.0
        ]

        # 重复填充到 64 维
        for i in range(8):
            obs[64 + i*8:64 + (i+1)*8] = workload_features

        # [128:192] 地理位置编码（one-hot + 延迟矩阵展平）
        # 简化为区域分布统计
        region_counts = {region: 0 for region in self.regions}
        for region in self.node_states['region']:
            if region in region_counts:
                region_counts[region] += 1

        region_features = [region_counts[region] / self.num_nodes for region in self.regions]
        # 重复填充到 64 维
        for i in range(64 // len(region_features)):
            start_idx = 128 + i * len(region_features)
            end_idx = min(128 + (i + 1) * len(region_features), 192)
            obs[start_idx:end_idx] = region_features[:end_idx - start_idx]

        # [192:224] 时间特征编码（小时、星期的正弦/余弦编码）
        # 模拟当前时间特征
        hour = (self.current_step // 3600) % 24  # 假设每步代表1秒
        day_of_week = (self.current_step // (3600 * 24)) % 7

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)

        time_features = [hour_sin, hour_cos, dow_sin, dow_cos]
        # 重复填充到 32 维
        for i in range(8):
            obs[192 + i*4:192 + (i+1)*4] = time_features

        # [224:256] 全局统计（平均队列长度、总待处理请求数等）
        global_features = [
            self.global_info.get('total_requests', 1000) / 100000.0,
            self.global_info.get('avg_input_tokens', 1000) / 4096.0,
            self.global_info.get('avg_output_tokens', 500) / 2048.0,
            self.global_info.get('avg_sla_deadline', 200) / 1500.0
        ]
        # 重复填充到 32 维
        for i in range(8):
            obs[224 + i*4:224 + (i+1)*4] = global_features

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一个动作并返回结果。

        Args:
            action: 选择的节点 ID（0 到 num_nodes-1）

        Returns:
            observation: 新的观测值
            reward: 获得的奖励
            terminated: 是否 episode 结束
            truncated: 是否因时间限制而截断
            info: 附加信息字典
        """
        # 验证动作有效性
        if action < 0 or action >= self.num_nodes:
            raise ValueError(f"Invalid node id: {action}. Must be in [0, {self.num_nodes-1}]")

        # 验证动作安全性（防止 OOM）
        is_safe = validate_action_safety(action, self.node_states, self.workload_queue)

        # 模拟执行动作后的状态变化
        info = {
            'completed_requests': 1 if is_safe else 0,
            'avg_latency': np.random.uniform(50, 500),
            'target_latency': self.workload_queue['sla_deadline_ms'],
            'avg_utilization': np.random.uniform(0.4, 0.9),
            'operational_cost': self.node_states['cost_per_hour'][action] / 3600.0,  # 每秒成本
            'revenue': 0.01,  # 假设每次请求收入
            'sla_compliance_rate': 1.0 if is_safe and np.random.uniform(50, 500) <= self.workload_queue['sla_deadline_ms'] else 0.0,
            'oom_occurred': not is_safe,
            'queue_overflow': False
        }

        # 计算奖励
        reward = calculate_reward({}, {}, action, info, {'reward_weights': self.reward_weights})

        # 更新内部状态
        self.current_step += 1
        self.episode_reward += reward

        # 更新工作负载队列（模拟新请求）
        if np.random.random() < 0.3:  # 30% 概率更新工作负载
            self.workload_queue = {
                'model_type': np.random.choice(['LLaMA3-7B', 'Qwen-14B', 'DeepSeek-70B']),
                'input_tokens': np.random.randint(50, 4096),
                'output_tokens': np.random.randint(50, 2048),
                'priority': np.random.choice(['high', 'medium', 'low'], p=[0.1, 0.3, 0.6]),
                'sla_deadline_ms': np.random.choice([100, 200, 400, 800])
            }

        # 编码新的观测值
        observation = self._encode_observation()

        # 检查 episode 是否结束
        terminated = False
        truncated = self.current_step >= 1000  # 最大步数限制

        return observation, reward, terminated, truncated, info

    def render(self):
        """渲染环境（可选）。"""
        pass

    def close(self):
        """关闭环境（可选）。"""
        pass
"""
环境测试模块：测试 RL 环境的正确性。
"""
import sys
import os
import pytest
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.envs.cluster_env import LLMClusterEnv
from src.envs.reward_fn import calculate_reward, validate_action_safety


class TestLLMClusterEnv:
    """LLMClusterEnv 的单元测试类。"""

    @pytest.fixture
    def env(self):
        """创建测试环境实例。"""
        env = LLMClusterEnv(num_nodes=10)
        return env

    def test_init(self, env):
        """测试环境初始化。"""
        assert env.num_nodes == 10
        assert env.observation_space.shape == (256,)
        assert env.action_space.n == 10
        assert len(env.regions) == 3

    def test_reset_returns_valid_observation(self, env):
        """测试 reset 返回的观测值格式正确。"""
        obs, info = env.reset(seed=42)

        # 检查观测值类型和形状
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (256,)
        assert obs.dtype == np.float32

        # 检查 info 字典
        assert 'node_states' in info
        assert 'workload_queue' in info
        assert 'global_info' in info

    def test_step_invalid_action_raises(self, env):
        """测试无效动作应抛出异常。"""
        env.reset(seed=42)

        # 动作超出范围应抛出 ValueError
        with pytest.raises(ValueError, match="Invalid node id"):
            env.step(-1)

        with pytest.raises(ValueError, match="Invalid node id"):
            env.step(100)  # 超出节点数量

    def test_step_valid_action(self, env):
        """测试有效动作的 step 执行。"""
        obs, _ = env.reset(seed=42)

        # 执行一个有效动作
        action = 0  # 选择第一个节点
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 检查返回值类型
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == (256,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_termination(self, env):
        """测试 episode 结束条件。"""
        env.reset(seed=42)

        # 运行直到 episode 结束
        total_steps = 0
        max_steps = 1001  # 环境默认最大 1000 步

        for _ in range(max_steps):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            total_steps += 1

            if terminated or truncated:
                break

        # episode 应该在 1000 步时结束
        assert total_steps <= 1000


class TestRewardFunction:
    """奖励函数的单元测试类。"""

    def test_calculate_reward_normal_case(self):
        """测试正常情况下的奖励计算。"""
        prev_state = {}
        current_state = {}
        action = 0
        info = {
            'completed_requests': 1,
            'avg_latency': 150,
            'target_latency': 200,
            'avg_utilization': 0.7,
            'operational_cost': 0.5,
            'revenue': 1.0,
            'sla_compliance_rate': 1.0,
            'oom_occurred': False,
            'queue_overflow': False
        }
        config = {
            'reward_weights': {
                'throughput': 0.3,
                'latency': 0.3,
                'utilization': 0.2,
                'cost': 0.1,
                'sla': 0.1
            }
        }

        reward = calculate_reward(prev_state, current_state, action, info, config)

        # 奖励应该是一个有限值
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert not np.isinf(reward)

    def test_calculate_reward_oom_penalty(self):
        """测试 OOM 惩罚。"""
        prev_state = {}
        current_state = {}
        action = 0
        info = {
            'completed_requests': 0,
            'avg_latency': 0,
            'target_latency': 200,
            'avg_utilization': 0.0,
            'operational_cost': 0.0,
            'revenue': 1.0,
            'sla_compliance_rate': 0.0,
            'oom_occurred': True,
            'queue_overflow': False
        }
        config = {}

        reward = calculate_reward(prev_state, current_state, action, info, config)

        # OOM 应该导致很大的负奖励
        assert reward < -5.0  # -10 惩罚

    def test_validate_action_safety(self):
        """测试动作安全性验证。"""
        # 安全的场景
        node_states = {
            'gpu_mem_gb': np.array([80, 48, 24])
        }
        workload_info = {
            'model_type': 'LLaMA3-7B',
            'input_tokens': 1000,
            'output_tokens': 500
        }

        # LLaMA3-7B 需要约 17GB 显存 (14GB base + 1500*2KB)
        # 80GB 节点是安全的
        assert validate_action_safety(0, node_states, workload_info) == True

        # 24GB 节点也是安全的
        assert validate_action_safety(2, node_states, workload_info) == True

    def test_validate_action_safety_oom(self):
        """测试不安全的动作（会导致 OOM）。"""
        node_states = {
            'gpu_mem_gb': np.array([24])  # 只有 24GB
        }
        workload_info = {
            'model_type': 'DeepSeek-70B',  # 大模型
            'input_tokens': 4000,
            'output_tokens': 2000
        }

        # DeepSeek-70B 需要约 236GB 显存 (140GB base + 6000*16KB)
        # 24GB 节点会触发 OOM
        assert validate_action_safety(0, node_states, workload_info) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
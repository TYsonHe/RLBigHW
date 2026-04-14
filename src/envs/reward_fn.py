"""
奖励函数模块：实现多目标加权奖励计算。
"""
import numpy as np
from typing import Dict, Any


def calculate_reward(
    prev_state: Dict[str, Any],
    current_state: Dict[str, Any],
    action: int,
    info: Dict[str, Any],
    config: Dict[str, Any]
) -> float:
    """
    计算多目标加权奖励。

    Args:
        prev_state: 上一时刻状态
        current_state: 当前时刻状态
        action: 执行的动作
        info: 包含详细信息的字典
        config: 奖励权重配置

    Returns:
        标量奖励值

    奖励组成：
    1. 基础奖励：吞吐量、延迟、利用率
    2. 约束惩罚：SLA 违规、资源溢出
    3. 稀疏奖励：episode 结束时的总评估
    """
    weights = config.get('reward_weights', {
        'throughput': 0.3,
        'latency': 0.3,
        'utilization': 0.2,
        'cost': 0.1,
        'sla': 0.1
    })

    # 1. 吞吐量奖励（已完成请求数）
    throughput_reward = info.get('completed_requests', 0) * 0.01

    # 2. 延迟惩罚（基于平均延迟与 SLA 的比率）
    avg_latency = info.get('avg_latency', 0)
    target_latency = info.get('target_latency', 200)
    if target_latency > 0:
        latency_ratio = avg_latency / target_latency
        latency_reward = -np.power(latency_ratio, 2)
    else:
        latency_reward = 0.0

    # 3. 资源利用率奖励（目标：60-80% 利用率）
    utilization = info.get('avg_utilization', 0)
    target_util = 0.7
    utilization_reward = 1.0 - abs(target_util - utilization)

    # 4. 成本惩罚
    cost = info.get('operational_cost', 0)
    revenue = info.get('revenue', 1)
    if revenue > 0:
        cost_reward = -(cost / revenue)
    else:
        cost_reward = -1.0

    # 5. SLA 达成奖励
    sla_rate = info.get('sla_compliance_rate', 1.0)
    sla_reward = sla_rate

    # 加权求和
    total_reward = (
        weights['throughput'] * throughput_reward +
        weights['latency'] * latency_reward +
        weights['utilization'] * utilization_reward +
        weights['cost'] * cost_reward +
        weights['sla'] * sla_reward
    )

    # 约束惩罚
    if info.get('oom_occurred', False):
        total_reward -= 10.0
    if info.get('queue_overflow', False):
        total_reward -= 5.0
    if sla_rate < 0.95:
        total_reward -= 2.0

    return total_reward


def validate_action_safety(
    action: int,
    node_states: Dict[str, Any],
    workload_info: Dict[str, Any]
) -> bool:
    """
    验证动作的安全性，防止 OOM 等问题。

    Args:
        action: 动作索引（节点 ID）
        node_states: 节点状态信息
        workload_info: 工作负载信息

    Returns:
        是否安全
    """
    if action >= len(node_states['gpu_mem_gb']) or action < 0:
        return False

    # 获取目标节点的 GPU 显存
    node_gpu_mem = node_states['gpu_mem_gb'][action]

    # 获取当前请求的模型类型和 token 长度
    model_type = workload_info.get('model_type', 'LLaMA3-7B')
    input_tokens = workload_info.get('input_tokens', 100)
    output_tokens = workload_info.get('output_tokens', 100)
    total_tokens = input_tokens + output_tokens

    # 根据模型类型计算所需显存
    model_vram_base = {
        'LLaMA3-7B': 14,   # GB
        'Qwen-14B': 28,    # GB
        'DeepSeek-70B': 140  # GB
    }

    kv_cache_per_token = {
        'LLaMA3-7B': 2 / 1024,   # GB per token (2KB)
        'Qwen-14B': 4 / 1024,    # GB per token (4KB)
        'DeepSeek-70B': 16 / 1024  # GB per token (16KB)
    }

    base_vram = model_vram_base.get(model_type, 14)
    kv_cache_vram = kv_cache_per_token.get(model_type, 2/1024) * total_tokens
    required_vram = base_vram + kv_cache_vram

    # 检查是否显存不足
    if required_vram > node_gpu_mem:
        return False

    return True
"""
工作负载流数据生成器：生成符合真实请求模式的工作负载数据。
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime, timedelta


class WorkloadStreamGenerator:
    """工作负载流数据生成器。"""

    def __init__(self):
        # 定义支持的模型类型（根据项目规范）
        self.model_types = ["LLaMA3-7B", "Qwen-14B", "DeepSeek-70B"]

        # Token 长度分布参数（对数正态分布）
        self.token_dist_params = {
            'input': {'mu': 7.5, 'sigma': 1.2, 'max': 4096, 'min': 50},
            'output': {'mu': 6.0, 'sigma': 0.8, 'max': 2048, 'min': 50}
        }

        # 请求优先级分布
        self.priority_probs = [0.1, 0.3, 0.6]  # 高:中:低 = 1:3:6

        # SLA 延迟约束（毫秒），按模型类型和优先级
        self.sla_constraints = {
            "LLaMA3-7B": {"high": 100, "medium": 200, "low": 500},
            "Qwen-14B": {"high": 200, "medium": 400, "low": 800},
            "DeepSeek-70B": {"high": 400, "medium": 800, "low": 1500}
        }

    def generate_token_lengths(self, num_samples: int, dist_type: str) -> np.ndarray:
        """生成符合长尾分布的 token 长度。

        Args:
            num_samples: 样本数量
            dist_type: 分布类型 ('input' 或 'output')

        Returns:
            token 长度数组
        """
        params = self.token_dist_params[dist_type]
        samples = np.random.lognormal(
            mean=params['mu'],
            sigma=params['sigma'],
            size=num_samples
        )
        return np.clip(samples, params['min'], params['max']).astype(int)

    def generate_arrival_times(self, duration_hours: float, lambda_base: float) -> List[float]:
        """生成符合潮汐效应的请求到达时间序列。

        使用非齐次泊松过程模拟请求到达，包含早9点和晚8点的流量高峰。

        Args:
            duration_hours: 模拟持续时间（小时）
            lambda_base: 基础到达率（请求/秒）

        Returns:
            到达时间列表（秒）
        """
        arrivals = []
        t = 0.0
        total_seconds = duration_hours * 3600

        while t < total_seconds:
            # 计算当前时刻的小时（用于潮汐效应）
            hour_of_day = (t / 3600) % 24

            # 基础潮汐效应：早9点和晚8点双峰
            tide_factor = (
                1.0 +
                0.5 * np.sin(2 * np.pi * (hour_of_day - 9) / 24) +  # 早9点高峰
                0.3 * np.sin(2 * np.pi * (hour_of_day - 20) / 24)   # 晚8点高峰
            )

            # 确保潮汐因子不低于基础值
            tide_factor = max(tide_factor, 0.5)

            # 当前时刻的瞬时到达率
            lambda_t = lambda_base * tide_factor

            # 生成下一个到达的时间间隔（指数分布）
            if lambda_t > 0:
                delta_t = np.random.exponential(1 / lambda_t)
                t += delta_t
                if t <= total_seconds:
                    arrivals.append(t)
            else:
                break

        return arrivals

    def generate_workload_stream(self,
                               num_requests: int,
                               duration_hours: float = 24.0,
                               base_rate: float = 2.0) -> pd.DataFrame:
        """生成工作负载流数据。

        Args:
            num_requests: 期望生成的请求数量
            duration_hours: 模拟持续时间（小时）
            base_rate: 基础请求到达率（请求/秒）

        Returns:
            包含工作负载信息的 DataFrame
        """
        # 生成到达时间
        arrival_times = self.generate_arrival_times(duration_hours, base_rate)

        # 如果生成的请求数量不够，调整基础率重新生成
        attempts = 0
        while len(arrival_times) < num_requests and attempts < 5:
            base_rate *= 1.5
            arrival_times = self.generate_arrival_times(duration_hours, base_rate)
            attempts += 1

        # 如果还是不够，截取或填充到目标数量
        if len(arrival_times) > num_requests:
            arrival_times = arrival_times[:num_requests]
        elif len(arrival_times) < num_requests:
            # 补充剩余的请求（均匀分布）
            remaining = num_requests - len(arrival_times)
            additional_times = np.random.uniform(
                0, duration_hours * 3600, remaining
            ).tolist()
            arrival_times.extend(additional_times)
            arrival_times.sort()

        # 生成请求特征
        model_types = np.random.choice(self.model_types, num_requests)
        input_tokens = self.generate_token_lengths(num_requests, 'input')
        output_tokens = self.generate_token_lengths(num_requests, 'output')
        priorities = np.random.choice(
            ['high', 'medium', 'low'],
            num_requests,
            p=self.priority_probs
        )

        # 根据模型类型和优先级设置 SLA 约束
        sla_deadlines = []
        for i in range(num_requests):
            model = model_types[i]
            priority = priorities[i]
            sla_deadline = self.sla_constraints[model][priority]
            sla_deadlines.append(sla_deadline)

        # 创建 DataFrame
        workload_data = {
            "timestamp": arrival_times,
            "req_id": [f"req_{i:06d}" for i in range(num_requests)],
            "model_type": model_types,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "priority": priorities,
            "sla_deadline_ms": sla_deadlines
        }

        df = pd.DataFrame(workload_data)

        # 对于测试集，添加突发流量场景
        if num_requests >= 10000:  # 测试集规模
            # 在随机位置插入突发流量（短时间内大量请求）
            burst_start_idx = num_requests // 2
            burst_size = min(1000, num_requests // 10)
            burst_duration = 60  # 60秒内的突发流量

            # 调整突发区域的时间戳
            if burst_start_idx + burst_size < num_requests:
                base_time = df.loc[burst_start_idx, "timestamp"]
                for i in range(burst_start_idx, burst_start_idx + burst_size):
                    df.loc[i, "timestamp"] = base_time + np.random.uniform(0, burst_duration)

                # 重新排序
                df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        """保存 DataFrame 到 CSV 文件。"""
        df.to_csv(output_path, index=False, encoding='utf-8')

    def generate_all_splits(self, output_dir: str):
        """生成所有数据集划分并保存到指定目录。"""
        os.makedirs(output_dir, exist_ok=True)

        # 根据项目规范生成不同规模的数据集
        splits_config = {
            "train": {"num_requests": 50000, "duration_hours": 168, "base_rate": 2.0},  # 7天
            "val": {"num_requests": 10000, "duration_hours": 24, "base_rate": 2.0},    # 1天
            "test": {"num_requests": 10000, "duration_hours": 24, "base_rate": 2.0}    # 1天 + 突发流量
        }

        for split, config in splits_config.items():
            print(f"Generating {split} workload streams ({config['num_requests']} requests)...")
            df = self.generate_workload_stream(**config)
            output_path = os.path.join(output_dir, f"workload_streams_{split}.csv")
            self.save_to_csv(df, output_path)
            print(f"Saved to {output_path}")
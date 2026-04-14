"""
集群拓扑数据生成器：生成符合真实硬件约束的节点配置。
"""
import os
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class ClusterTopologyGenerator:
    """集群拓扑数据生成器。"""

    def __init__(self):
        # 定义支持的硬件配置（根据项目规范）
        self.gpu_specs = {
            "NVIDIA A100-SXM4-80GB": {
                "gpu_mem_gb": 80,
                "cost_per_hour": 3.5,
                "network_bandwidth_mbps": 20000
            },
            "NVIDIA L40S-48GB": {
                "gpu_mem_gb": 48,
                "cost_per_hour": 2.0,
                "network_bandwidth_mbps": 10000
            },
            "NVIDIA RTX 4090-24GB": {
                "gpu_mem_gb": 24,
                "cost_per_hour": 1.2,
                "network_bandwidth_mbps": 5000
            }
        }

        self.cpu_options = [64, 128, 256]  # CPU 核数选项
        self.mem_options = [256, 512, 1024]  # 内存 GB 选项
        self.disk_options = [2000, 4000, 8000]  # 磁盘 GB 选项
        self.regions = ["beijing", "shanghai", "guangzhou"]

        # 地理位置间的网络延迟矩阵（毫秒）
        self.region_latency_matrix = {
            ("beijing", "beijing"): 0,
            ("beijing", "shanghai"): 20,
            ("beijing", "guangzhou"): 35,
            ("shanghai", "beijing"): 20,
            ("shanghai", "shanghai"): 0,
            ("shanghai", "guangzhou"): 25,
            ("guangzhou", "beijing"): 35,
            ("guangzhou", "shanghai"): 25,
            ("guangzhou", "guangzhou"): 0
        }

    def generate(self, num_nodes: int, split: str) -> pd.DataFrame:
        """生成指定数量的节点配置。

        Args:
            num_nodes: 节点数量
            split: 数据集划分类型 ('train', 'val', 'test')

        Returns:
            包含节点配置的 DataFrame
        """
        nodes_data = []

        for i in range(num_nodes):
            # 随机选择硬件配置
            gpu_model = random.choice(list(self.gpu_specs.keys()))
            gpu_info = self.gpu_specs[gpu_model]

            node_config = {
                "node_id": f"node_{split}_{i:04d}",
                "region": random.choice(self.regions),
                "cpu_cores": random.choice(self.cpu_options),
                "mem_gb": random.choice(self.mem_options),
                "disk_gb": random.choice(self.disk_options),
                "gpu_model": gpu_model,
                "gpu_mem_gb": gpu_info["gpu_mem_gb"],
                "cost_per_hour": gpu_info["cost_per_hour"],
                "network_bandwidth_mbps": gpu_info["network_bandwidth_mbps"]
            }
            nodes_data.append(node_config)

        df = pd.DataFrame(nodes_data)

        # 对于测试集，添加一些极端场景
        if split == "test":
            # 添加单一 GPU 型号的场景
            single_gpu_nodes = min(10, num_nodes // 2)
            for i in range(single_gpu_nodes):
                df.loc[i, "gpu_model"] = "NVIDIA A100-SXM4-80GB"
                df.loc[i, "gpu_mem_gb"] = 80
                df.loc[i, "cost_per_hour"] = 3.5
                df.loc[i, "network_bandwidth_mbps"] = 20000

            # 添加跨地域高延迟场景
            cross_region_nodes = min(10, num_nodes // 2)
            for i in range(single_gpu_nodes, single_gpu_nodes + cross_region_nodes):
                df.loc[i, "region"] = "beijing"

        return df

    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        """保存 DataFrame 到 CSV 文件。"""
        df.to_csv(output_path, index=False, encoding='utf-8')

    def generate_all_splits(self, output_dir: str):
        """生成所有数据集划分并保存到指定目录。"""
        os.makedirs(output_dir, exist_ok=True)

        # 根据项目规范生成不同规模的数据集
        splits_config = {
            "train": 500,
            "val": 100,
            "test": 50
        }

        for split, num_nodes in splits_config.items():
            print(f"Generating {split} cluster profiles ({num_nodes} nodes)...")
            df = self.generate(num_nodes, split)
            output_path = os.path.join(output_dir, f"cluster_profiles_{split}.csv")
            self.save_to_csv(df, output_path)
            print(f"Saved to {output_path}")
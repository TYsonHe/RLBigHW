"""
数据加载与预处理工具：提供数据加载管道。
"""
import os
import pandas as pd
from typing import Iterator, Dict, Any


class DataPipeline:
    """数据加载管道。"""

    def __init__(self, data_dir: str = "Datasets/generated"):
        """初始化数据管道。

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.cluster_data = {}
        self.workload_data = {}

        # 预加载集群拓扑数据（相对较小）
        for split in ["train", "val", "test"]:
            cluster_file = os.path.join(data_dir, f"cluster_profiles_{split}.csv")
            if os.path.exists(cluster_file):
                self.cluster_data[split] = pd.read_csv(cluster_file)

    def load_cluster(self, split: str) -> pd.DataFrame:
        """加载指定划分的集群拓扑数据。

        Args:
            split: 数据集划分 ('train', 'val', 'test')

        Returns:
            集群拓扑 DataFrame
        """
        if split not in self.cluster_data:
            cluster_file = os.path.join(self.data_dir, f"cluster_profiles_{split}.csv")
            if os.path.exists(cluster_file):
                self.cluster_data[split] = pd.read_csv(cluster_file)
            else:
                raise FileNotFoundError(f"Cluster profile file not found: {cluster_file}")

        return self.cluster_data[split].copy()

    def get_workload_iterator(self, split: str, batch_size: int = 1000) -> Iterator[pd.DataFrame]:
        """获取工作负载数据的迭代器。

        由于工作负载数据量大，使用迭代器按批次从磁盘读取。

        Args:
            split: 数据集划分 ('train', 'val', 'test')
            batch_size: 批次大小

        Yields:
            工作负载批次 DataFrame
        """
        workload_file = os.path.join(self.data_dir, f"workload_streams_{split}.csv")
        if not os.path.exists(workload_file):
            raise FileNotFoundError(f"Workload stream file not found: {workload_file}")

        # 使用 chunksize 参数进行分块读取
        for chunk in pd.read_csv(workload_file, chunksize=batch_size):
            yield chunk

    def get_global_info(self, split: str) -> Dict[str, Any]:
        """获取全局信息（如平均队列长度、总待处理请求数等）。

        Args:
            split: 数据集划分

        Returns:
            全局信息字典
        """
        workload_file = os.path.join(self.data_dir, f"workload_streams_{split}.csv")
        if not os.path.exists(workload_file):
            return {}

        df = pd.read_csv(workload_file)
        total_requests = len(df)
        avg_input_tokens = df["input_tokens"].mean()
        avg_output_tokens = df["output_tokens"].mean()
        avg_sla_deadline = df["sla_deadline_ms"].mean()

        return {
            "total_requests": total_requests,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "avg_sla_deadline": avg_sla_deadline,
            "model_distribution": df["model_type"].value_counts().to_dict(),
            "priority_distribution": df["priority"].value_counts().to_dict()
        }
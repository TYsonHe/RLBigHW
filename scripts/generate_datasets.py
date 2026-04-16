"""
数据集生成脚本：生成所有仿真数据集。
"""
import os
import sys
import argparse

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.data_generators import ClusterTopologyGenerator, WorkloadStreamGenerator


def generate_datasets(output_dir: str, seed: int = 42):
    """生成所有数据集。

    Args:
        output_dir: 输出目录
        seed: 随机种子
    """
    # 设置随机种子
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("开始生成仿真数据集")
    print("=" * 60)

    # 1. 生成集群拓扑数据
    print("\n[1/2] 生成集群拓扑数据...")
    cluster_generator = ClusterTopologyGenerator()
    cluster_generator.generate_all_splits(output_dir)

    # 2. 生成工作负载流数据
    print("\n[2/2] 生成工作负载流数据...")
    workload_generator = WorkloadStreamGenerator()
    workload_generator.generate_all_splits(output_dir)

    print("\n" + "=" * 60)
    print("数据集生成完成！")
    print("=" * 60)
    print(f"输出目录: {output_dir}")

    # 列出生成的文件
    import os
    generated_files = os.listdir(output_dir)
    print(f"\n生成的文件:")
    for f in sorted(generated_files):
        file_path = os.path.join(output_dir, f)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  - {f} ({file_size:.2f} KB)")


def main():
    parser = argparse.ArgumentParser(description="生成仿真数据集")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Datasets/generated",
        help="输出目录 (默认: Datasets/generated)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )

    args = parser.parse_args()

    generate_datasets(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
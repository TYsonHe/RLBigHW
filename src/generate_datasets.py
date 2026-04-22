# DQN/generate_datasets.py
import os
import sys

# 将 DQN 目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_generators.cluster_generator import main as gen_cluster
from utils.data_generators.workload_generator import main as gen_workload

if __name__ == "__main__":
    gen_cluster()
    gen_workload()
    print("\n✅ All datasets generated in DQN/data/")
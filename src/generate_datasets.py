import os
import sys

# 项目根目录是 src 的父目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入加 src 前缀
from src.utils.data_generators.cluster_generator import main as gen_cluster
from src.utils.data_generators.workload_generator import main as gen_workload

if __name__ == "__main__":
    gen_cluster()
    gen_workload()
    print(f"\n✅ All datasets saved to {os.path.join(PROJECT_ROOT, 'data')}")

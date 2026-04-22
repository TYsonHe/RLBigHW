# DQN/utils/data_generators/cluster_generator.py
import pandas as pd
import numpy as np
import os

GPU_SPECS = {
    "NVIDIA A100-SXM4-80GB": {"mem_gb": 80, "cost": 30.0, "flops": 312},
    "NVIDIA L40S-48GB": {"mem_gb": 48, "cost": 12.0, "flops": 180},
    "NVIDIA RTX 4090-24GB": {"mem_gb": 24, "cost": 5.0, "flops": 82},
}

REGIONS = ["beijing", "shanghai", "guangzhou"]

def generate_cluster(num_nodes: int, split: str, seed: int = 42):
    np.random.seed(seed + hash(split) % 1000)
    records = []
    
    gpu_types = list(GPU_SPECS.keys())
    weights = [0.2, 0.3, 0.5]
    
    for i in range(num_nodes):
        gpu = np.random.choice(gpu_types, p=weights)
        region = np.random.choice(REGIONS)
        spec = GPU_SPECS[gpu]
        
        cpu_cores = np.random.choice([32, 64, 128])
        mem_gb = np.random.choice([256, 512])
        disk_gb = np.random.choice([2000, 4000])
        
        base_bw = np.random.randint(10000, 100000)
        
        records.append({
            "node_id": f"node_{split}_{i:04d}",
            "region": region,
            "cpu_cores": f"{cpu_cores} cores",
            "mem_gb": f"{mem_gb} GB",
            "disk_gb": f"{disk_gb} GB NVMe",
            "gpu_model": gpu,
            "gpu_mem_gb": spec["mem_gb"],
            "cost_per_hour": round(spec["cost"] * (1 + np.random.uniform(-0.1, 0.1)), 2),
            "network_bandwidth_mbps": base_bw,
            "base_flops": spec["flops"],
        })
    
    return pd.DataFrame(records)

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(base_dir, exist_ok=True)
    
    configs = [("train", 500, 42), ("val", 100, 43), ("test", 50, 44)]
    
    for split, num, seed in configs:
        df = generate_cluster(num_nodes=num, split=split, seed=seed)
        df.to_csv(os.path.join(base_dir, f"cluster_profiles_{split}.csv"), index=False)
        print(f"[Cluster] {split}: {num} nodes generated -> DQN/data/")

if __name__ == "__main__":
    main()
# DQN/utils/data_generators/workload_generator.py
import pandas as pd
import numpy as np
import os

MODELS = {
    "LLaMA3-7B": {"weight_gb": 14, "kv_per_token_kb": 2, "ms_per_token": {"A100": 2.5, "L40S": 3.2, "4090": 4.0}},
    "Qwen-14B": {"weight_gb": 28, "kv_per_token_kb": 4, "ms_per_token": {"A100": 4.0, "L40S": 5.5, "4090": 8.0}},
    "DeepSeek-70B": {"weight_gb": 140, "kv_per_token_kb": 16, "ms_per_token": {"A100": 12.0, "L40S": 18.0, "4090": 9999.0}},
}

def generate_workload_stream(num_requests: int, duration_hours: float, split: str, seed: int = 42):
    np.random.seed(seed)
    duration_sec = duration_hours * 3600
    
    lambda_base = 50
    A = 0.5
    phi = 9.0
    T = 24.0
    
    arrivals = []
    t = 0.0
    while t < duration_sec and len(arrivals) < num_requests:
        hour_of_day = (t / 3600) % 24
        lambda_t = lambda_base * (1 + A * np.sin(2 * np.pi * (hour_of_day - phi) / T))
        lambda_t = max(lambda_t, 5.0)
        
        delta_t = np.random.exponential(1.0 / lambda_t)
        t += delta_t
        arrivals.append(t)
    
    def sample_tokens(n, dist_type):
        params = {
            'input': {'mu': 7.5, 'sigma': 1.2, 'max': 4096, 'min': 50},
            'output': {'mu': 6.0, 'sigma': 0.8, 'max': 2048, 'min': 50},
        }
        p = params[dist_type]
        samples = np.random.lognormal(mean=p['mu'], sigma=p['sigma'], size=n)
        return np.clip(samples, p['min'], p['max']).astype(int)
    
    n = len(arrivals)
    model_types = np.random.choice(list(MODELS.keys()), size=n, p=[0.5, 0.35, 0.15])
    input_lens = sample_tokens(n, 'input')
    output_lens = sample_tokens(n, 'output')
    
    def get_sla(model, in_len):
        base = 200 if model == "LLaMA3-7B" else (500 if model == "Qwen-14B" else 2000)
        return int(base * (1 + 0.1 * (in_len / 1000)))
    
    records = []
    for i in range(n):
        model = model_types[i]
        sla = get_sla(model, input_lens[i])
        records.append({
            "timestamp": round(arrivals[i], 3),
            "req_id": f"{split}_req_{i:06d}",
            "model_type": model,
            "input_tokens": int(input_lens[i]),
            "output_tokens": int(output_lens[i]),
            "priority": int(np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])),
            "sla_deadline_ms": sla,
        })
    
    df = pd.DataFrame(records)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(base_dir, exist_ok=True)
    
    configs = [("train", 50000, 24 * 7, 42), ("val", 10000, 24, 43), ("test", 10000, 24, 44)]
    
    for split, num, hours, seed in configs:
        df = generate_workload_stream(num, hours, split, seed)
        df.to_csv(os.path.join(base_dir, f"workload_streams_{split}.csv"), index=False)
        print(f"[Workload] {split}: {len(df)} requests over {hours}h generated -> DQN/data/")

if __name__ == "__main__":
    main()
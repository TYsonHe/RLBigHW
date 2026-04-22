# DQN/envs/cluster_env.py
import gymnasium as gym
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional

GPU_MODEL_MAP = {
    "NVIDIA A100-SXM4-80GB": "A100",
    "NVIDIA L40S-48GB": "L40S",
    "NVIDIA RTX 4090-24GB": "4090",
}

MODEL_SPECS = {
    "LLaMA3-7B": {"weight_gb": 14, "kv_per_token_kb": 2, "ms_per_token": {"A100": 2.5, "L40S": 3.2, "4090": 4.0}},
    "Qwen-14B": {"weight_gb": 28, "kv_per_token_kb": 4, "ms_per_token": {"A100": 4.0, "L40S": 5.5, "4090": 8.0}},
    "DeepSeek-70B": {"weight_gb": 140, "kv_per_token_kb": 16, "ms_per_token": {"A100": 12.0, "L40S": 18.0, "4090": 9999.0}},
}

POOL_NAMES = ["short", "long", "mixed"]

class LLMClusterEnv(gym.Env):
    """
    14维状态空间，36离散动作空间
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, cluster_csv: str, workload_csv: str, num_nodes: int = 100,
                 reward_weights: Optional[Dict] = None, step_batch: int = 10):
        super().__init__()
        
        self.step_batch = step_batch
        self.num_nodes_total = num_nodes
        
        self.cluster_df = pd.read_csv(cluster_csv).head(num_nodes)
        self.nodes = self._init_nodes()
        
        self.workload_df = pd.read_csv(workload_csv)
        self.workload_iter = self.workload_df.iterrows()
        self.workload_buffer: List[Dict] = []
        
        self.action_space = gym.spaces.Discrete(36)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        self.w = reward_weights or {
            "throughput": 0.3, "latency": 0.3, "utilization": 0.2,
            "cost": 0.1, "sla": 0.1
        }
        
        self.pool_cfg = {
            0: {"alpha": 0.4, "beta": 2, "gamma": 0.7, "v": 0.6, "m": 0.2, "c": 0.5},
            1: {"alpha": 0.4, "beta": 1, "gamma": 0.3, "v": 1.0, "m": 0.8, "c": 0.9},
            2: {"alpha": 0.2, "beta": 1, "gamma": 0.0, "v": 0.8, "m": 0.5, "c": 0.6},
        }
        
        self.current_time = 0.0
        self.total_reward = 0.0
        self.episode_len = 0
        self.stats = {"completed": 0, "oom": 0, "sla_violations": 0, "total_cost": 0.0}
    
    def _init_nodes(self):
        nodes = []
        for _, row in self.cluster_df.iterrows():
            gpu_short = GPU_MODEL_MAP[row["gpu_model"]]
            nodes.append({
                "id": row["node_id"],
                "region": row["region"],
                "gpu_type": gpu_short,
                "gpu_mem_total_gb": row["gpu_mem_gb"],
                "cost_per_hour": row["cost_per_hour"],
                "pool_id": -1,
                "active": False,
                "instances": 0,
                "queue": 0,
                "gpu_mem_util": 0.0,
                "cpu_util": 0.0,
                "processing": [],
            })
        return nodes
    
    def _decode_action(self, action_id: int) -> Dict:
        a = action_id // 12
        b = (action_id % 12) // 4
        c = action_id % 4
        
        gamma_patterns = [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.4, 0.4, 0.2],
        ]
        
        v_m_presets = [(0.4, 0.2), (0.6, 0.4), (0.8, 0.6), (1.0, 0.9)]
        v, m = v_m_presets[c]
        
        cfg = {}
        for g in range(3):
            base_beta = self.pool_cfg[g]["beta"]
            delta = -1 if b == 0 else (0 if b == 1 else 2)
            new_beta = max(1, min(20, base_beta + delta))
            
            cfg[g] = {
                "alpha": self.pool_cfg[g]["alpha"],
                "beta": new_beta,
                "gamma": gamma_patterns[a][g],
                "v": v if g != 0 else min(v, 0.8),
                "m": m,
                "c": 0.5 + 0.4 * (c / 3),
            }
        return cfg
    
    def _apply_pool_reconfig(self, target_cfg: Dict):
        sorted_nodes = sorted(self.nodes, key=lambda n: n["cost_per_hour"])
        
        n_active = sum(1 for n in self.nodes if n["active"])
        if n_active == 0:
            n_active = self.num_nodes_total
            
        pool_target_nodes = {}
        for g in range(3):
            pool_target_nodes[g] = max(1, int(target_cfg[g]["alpha"] * n_active))
        
        for node in self.nodes:
            node["pool_id"] = -1
            node["active"] = False
            node["instances"] = 0
        
        for g in range(3):
            count = 0
            for node in sorted_nodes:
                if node["pool_id"] != -1:
                    continue
                if count >= pool_target_nodes[g]:
                    break
                if g == 0 and node["gpu_type"] == "4090":
                    node["pool_id"] = g; node["active"] = True; count += 1
                elif g == 1 and node["gpu_type"] == "A100":
                    node["pool_id"] = g; node["active"] = True; count += 1
                elif g == 2:
                    node["pool_id"] = g; node["active"] = True; count += 1
        
        for node in self.nodes:
            if node["pool_id"] == -1:
                node["active"] = False; node["instances"] = 0; node["queue"] = 0
        
        for g in range(3):
            pool_nodes = [n for n in self.nodes if n["pool_id"] == g and n["active"]]
            if not pool_nodes:
                continue
            beta = target_cfg[g]["beta"]
            base = beta // len(pool_nodes)
            remainder = beta % len(pool_nodes)
            
            for idx, node in enumerate(pool_nodes):
                node["instances"] = base + (1 if idx < remainder else 0)
                max_inst_by_vram = int(1.0 // target_cfg[g]["v"]) if target_cfg[g]["v"] > 0 else 1
                node["instances"] = min(node["instances"], max(1, max_inst_by_vram))
                
                weight_gb = 20
                if node["instances"] > 0:
                    node["gpu_mem_util"] = (weight_gb * node["instances"]) / node["gpu_mem_total_gb"]
                else:
                    node["gpu_mem_util"] = 0.0
        
        for g in range(3):
            self.pool_cfg[g].update(target_cfg[g])
    
    def _admit_request(self, req: Dict) -> Tuple[bool, int, float]:
        p = np.array([self.pool_cfg[g]["gamma"] for g in range(3)])
        p /= p.sum()
        target_pool = int(np.random.choice(3, p=p))
        
        pool_nodes = [n for n in self.nodes if n["pool_id"] == target_pool and n["active"]]
        if not pool_nodes:
            return False, target_pool, -5.0
        
        model_spec = MODEL_SPECS[req["model_type"]]
        vram_req = model_spec["weight_gb"] + model_spec["kv_per_token_kb"] * (req["input_tokens"] + req.get("output_tokens", 0)) / 1024
        
        candidates = sorted(pool_nodes, key=lambda n: n["queue"])
        chosen = candidates[0]
        
        v_limit = self.pool_cfg[target_pool]["v"] * chosen["gpu_mem_total_gb"]
        current_used = chosen["gpu_mem_util"] * chosen["gpu_mem_total_gb"]
        
        if current_used + vram_req > v_limit:
            if self.pool_cfg[target_pool]["m"] > 0.5:
                req["_offload"] = True
                req["_latency_multiplier"] = 1.5 + (1.0 - self.pool_cfg[target_pool]["m"])
            else:
                return False, target_pool, -10.0
        
        chosen["queue"] += 1
        chosen["gpu_mem_util"] = min(1.0, (current_used + vram_req) / chosen["gpu_mem_total_gb"])
        req["_target_node"] = chosen["id"]
        req["_target_pool"] = target_pool
        req["_start_time"] = self.current_time
        return True, target_pool, 0.0
    
    def _process_requests(self, dt: float = 1.0):
        completed_this_step = 0
        latencies = []
        sla_violations = 0
        cost_this_step = 0.0
        
        for node in self.nodes:
            if not node["active"] or node["queue"] == 0:
                continue
            
            pool_id = node["pool_id"]
            cfg = self.pool_cfg[pool_id]
            
            proc_tokens_per_sec = 1000 / MODEL_SPECS["LLaMA3-7B"]["ms_per_token"].get(node["gpu_type"], 10.0)
            proc_tokens_per_sec *= node["instances"] * cfg["c"]
            
            process_count = min(node["queue"], max(1, int(proc_tokens_per_sec / 500)))
            node["queue"] -= process_count
            node["queue"] = max(0, node["queue"])
            node["gpu_mem_util"] = max(0.1, node["gpu_mem_util"] * 0.8)
            
            completed_this_step += process_count
            cost_this_step += node["cost_per_hour"] * (dt / 3600) * (1 if node["queue"] > 0 else 0.5)
            
            for _ in range(process_count):
                latency = np.random.uniform(50, 500) * (1.5 if cfg["m"] > 0.6 else 1.0)
                latencies.append(latency)
                if latency > 200:
                    sla_violations += 1
        
        self.stats["completed"] += completed_this_step
        self.stats["sla_violations"] += sla_violations
        self.stats["total_cost"] += cost_this_step
        
        return completed_this_step, latencies, sla_violations, cost_this_step
    
    def _get_observation(self) -> np.ndarray:
        state = np.zeros(14, dtype=np.float32)
        
        for g in range(3):
            pool_nodes = [n for n in self.nodes if n["pool_id"] == g and n["active"]]
            if pool_nodes:
                queues = [n["queue"] for n in pool_nodes]
                gpu_mems = [n["gpu_mem_util"] for n in pool_nodes]
                state[3*g] = max(queues) / 100.0
                state[3*g + 1] = np.mean(gpu_mems)
                state[3*g + 2] = len(pool_nodes) / self.num_nodes_total
            else:
                state[3*g : 3*g+3] = [0.0, 0.0, 0.0]
        
        if self.workload_buffer:
            wait_times = [self.current_time - req.get("timestamp", self.current_time) for req in self.workload_buffer]
            wait_times = [w for w in wait_times if w >= 0]
            if wait_times:
                state[9] = np.percentile(wait_times, 95) / 10.0
            
            total_lens = [req["input_tokens"] + req.get("output_tokens", 0) for req in self.workload_buffer]
            state[10] = sum(1 for l in total_lens if l > 8000) / len(self.workload_buffer)
            
            priorities = [req.get("priority", 2) for req in self.workload_buffer]
            state[11] = sum(1 for p in priorities if p == 1) / len(self.workload_buffer)
        else:
            state[9:12] = [0.0, 0.0, 0.0]
        
        hour = (self.current_time / 3600) % 24
        rad = 2 * math.pi * hour / 24
        state[12] = math.sin(rad)
        state[13] = math.cos(rad)
        
        return state
    
    def _compute_reward(self, completed, latencies, sla_violations, cost, penalty) -> float:
        R_tp = completed / 10.0
        R_lat = -np.mean(latencies) / 500.0 if latencies else 0.0
        
        active_nodes = [n for n in self.nodes if n["active"]]
        R_util = 0.0
        if active_nodes:
            utils = [n["gpu_mem_util"] for n in active_nodes]
            R_util = -np.mean([abs(u - 0.7) for u in utils])
        
        R_cost = -cost / 10.0
        R_sla = -sla_violations
        
        reward = (self.w["throughput"] * R_tp +
                  self.w["latency"] * R_lat +
                  self.w["utilization"] * R_util +
                  self.w["cost"] * R_cost +
                  self.w["sla"] * R_sla +
                  penalty)
        return float(reward)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        self.total_reward = 0.0
        self.episode_len = 0
        self.stats = {"completed": 0, "oom": 0, "sla_violations": 0, "total_cost": 0.0}
        self.workload_iter = self.workload_df.iterrows()
        self.workload_buffer = []
        
        for node in self.nodes:
            node["pool_id"] = -1; node["active"] = False; node["instances"] = 0
            node["queue"] = 0; node["gpu_mem_util"] = 0.0
        
        self.pool_cfg = {
            0: {"alpha": 0.4, "beta": 2, "gamma": 0.7, "v": 0.6, "m": 0.2, "c": 0.5},
            1: {"alpha": 0.4, "beta": 1, "gamma": 0.3, "v": 1.0, "m": 0.8, "c": 0.9},
            2: {"alpha": 0.2, "beta": 1, "gamma": 0.0, "v": 0.8, "m": 0.5, "c": 0.6},
        }
        
        self._apply_pool_reconfig(self.pool_cfg)
        self._arrive_requests()
        
        return self._get_observation(), {}
    
    def _arrive_requests(self):
        try:
            for _ in range(self.step_batch):
                idx, row = next(self.workload_iter)
                self.workload_buffer.append(row.to_dict())
                self.current_time = max(self.current_time, row["timestamp"])
        except StopIteration:
            pass
    
    def step(self, action: int):
        self.episode_len += 1
        
        target_cfg = self._decode_action(action)
        self._apply_pool_reconfig(target_cfg)
        
        self._arrive_requests()
        
        penalty = 0.0
        admitted = 0
        new_buffer = []
        for req in self.workload_buffer:
            ok, pool, p = self._admit_request(req)
            if ok:
                admitted += 1
            else:
                new_buffer.append(req)
                penalty += p
                if p <= -10:
                    self.stats["oom"] += 1
        
        self.workload_buffer = new_buffer
        
        completed, latencies, sla_violations, cost = self._process_requests()
        reward = self._compute_reward(completed, latencies, sla_violations, cost, penalty)
        self.total_reward += reward
        
        terminated = len(self.workload_buffer) == 0 and self.episode_len > 1000
        truncated = self.episode_len > 5000
        
        info = {
            "completed": completed,
            "admitted": admitted,
            "penalty": penalty,
            "sla_violations": sla_violations,
            "cost": cost,
        }
        
        return self._get_observation(), reward, terminated, truncated, info
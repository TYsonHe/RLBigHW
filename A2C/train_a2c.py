# A2C/train_a2c.py
"""A2C 训练脚本：支持训练/验证/测试，日志带时间戳。"""
import os
import sys
import json
import yaml
import numpy as np
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.envs.cluster_env import LLMClusterEnv
from algorithms.a2c import A2CAgent

SEED = 42


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cluster_csv: str, workload_csv: str, cfg: dict) -> LLMClusterEnv:
    env_cfg = cfg.get("env", {})
    return LLMClusterEnv(
        cluster_csv=cluster_csv,
        workload_csv=workload_csv,
        num_nodes=env_cfg.get("num_nodes", 100),
        reward_weights=env_cfg.get("reward_weights"),
        step_batch=env_cfg.get("step_batch", 10),
    )


def run_episode(env: LLMClusterEnv, agent: A2CAgent,
                max_steps: int, n_steps: int,
                deterministic: bool = False,
                seed: int = None) -> dict:
    """运行一个 episode，每 n_steps 执行一次 A2C 更新。

    Returns:
        包含 episode 统计信息的字典
    """
    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_steps = 0
    update_losses = []

    for t in range(max_steps):
        result = agent.select_action(state, deterministic=deterministic)
        action = result["action"]
        log_prob = result["log_prob"]
        value = result["value"]

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.buffer.add(state, action, reward, value, log_prob, float(done))
        episode_reward += reward
        episode_steps += 1

        # 每 n_steps 或 episode 结束时更新
        if (t + 1) % n_steps == 0 or done:
            with torch.no_grad():
                _, next_val = agent.network(
                    torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                )
            nv = next_val.item() if not done else 0.0
            loss_dict = agent.learn(nv)
            if loss_dict:
                update_losses.append(loss_dict)

        state = next_state
        if done:
            break

    # 汇总损失
    avg_losses = {}
    if update_losses:
        for key in update_losses[0]:
            avg_losses[key] = float(np.mean([d[key] for d in update_losses]))

    # 获取 agent 内部记录的 episode 损失
    ep_losses = agent.get_episode_loss_avg()

    return {
        "reward": round(episode_reward, 2),
        "steps": episode_steps,
        "completed": env.stats["completed"],
        "oom": env.stats["oom"],
        "sla_violations": env.stats["sla_violations"],
        "total_cost": round(env.stats["total_cost"], 4),
        "policy_loss": ep_losses.get("policy_loss"),
        "value_loss": ep_losses.get("value_loss"),
        "entropy": ep_losses.get("entropy"),
    }


def evaluate(env: LLMClusterEnv, agent: A2CAgent, cfg: dict,
             num_episodes: int, seed_base: int = 10000) -> dict:
    """确定性策略评估。"""
    train_cfg = cfg.get("training", {})
    max_steps = train_cfg.get("max_steps", 500)
    n_steps = train_cfg.get("n_steps", 10)

    all_rewards = []
    all_stats = {"completed": [], "oom": [], "sla_violations": [], "total_cost": []}

    for i in range(num_episodes):
        result = run_episode(env, agent, max_steps, n_steps,
                             deterministic=True, seed=seed_base + i)
        all_rewards.append(result["reward"])
        for k in all_stats:
            all_stats[k].append(result[k])

    summary = {
        "reward_mean": round(float(np.mean(all_rewards)), 2),
        "reward_std": round(float(np.std(all_rewards)), 2),
        "completed_mean": round(float(np.mean(all_stats["completed"])), 1),
        "oom_mean": round(float(np.mean(all_stats["oom"])), 1),
        "sla_violations_mean": round(float(np.mean(all_stats["sla_violations"])), 1),
        "total_cost_mean": round(float(np.mean(all_stats["total_cost"])), 4),
    }
    return summary


def train():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[A2C] 训练开始，时间戳: {timestamp}")

    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    cfg = load_config(config_path)

    # 固定随机种子
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    a2c_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(a2c_dir, "checkpoints")
    log_dir = os.path.join(a2c_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 保存本次训练的超参数快照
    config_snapshot = os.path.join(log_dir, f"config_{timestamp}.yaml")
    with open(config_snapshot, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # 创建环境
    train_env = make_env(
        os.path.join(data_dir, "cluster_profiles_train.csv"),
        os.path.join(data_dir, "workload_streams_train.csv"),
        cfg,
    )
    val_env = make_env(
        os.path.join(data_dir, "cluster_profiles_val.csv"),
        os.path.join(data_dir, "workload_streams_val.csv"),
        cfg,
    )
    test_env = make_env(
        os.path.join(data_dir, "cluster_profiles_test.csv"),
        os.path.join(data_dir, "workload_streams_test.csv"),
        cfg,
    )

    # 创建 Agent
    agent_cfg = cfg.get("agent", {})
    agent = A2CAgent(
        state_dim=agent_cfg.get("state_dim", 14),
        action_dim=agent_cfg.get("action_dim", 36),
        hidden_dims=agent_cfg.get("hidden_dims", [256, 256]),
        lr=agent_cfg.get("lr", 7e-4),
        gamma=agent_cfg.get("gamma", 0.99),
        gae_lambda=agent_cfg.get("gae_lambda", 0.95),
        entropy_coef=agent_cfg.get("entropy_coef", 0.01),
        value_coef=agent_cfg.get("value_coef", 0.5),
        max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
    )

    # 日志文件
    train_log_path = os.path.join(log_dir, f"train_{timestamp}.jsonl")
    val_log_path = os.path.join(log_dir, f"val_{timestamp}.jsonl")
    train_log = open(train_log_path, "w", encoding="utf-8")
    val_log = open(val_log_path, "w", encoding="utf-8")

    train_cfg = cfg.get("training", {})
    num_episodes = train_cfg.get("num_episodes", 2000)
    max_steps = train_cfg.get("max_steps", 500)
    n_steps = train_cfg.get("n_steps", 10)
    val_interval = train_cfg.get("val_interval", 100)
    val_episodes = train_cfg.get("val_episodes", 5)
    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("save_interval", 200)

    best_val_reward = -float("inf")

    for episode in range(num_episodes):
        result = run_episode(
            train_env, agent, max_steps, n_steps,
            deterministic=False, seed=SEED + episode,
        )

        # 写训练日志
        log_entry = {"timestamp": timestamp, "episode": episode}
        log_entry.update(result)
        train_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        train_log.flush()

        if episode % log_interval == 0:
            print(
                f"[Train] Ep {episode:4d} | Reward: {result['reward']:8.2f} | "
                f"Completed: {result['completed']} | OOM: {result['oom']} | "
                f"SLA Viol: {result['sla_violations']} | "
                f"PolicyLoss: {result['policy_loss']} | "
                f"Entropy: {result['entropy']}"
            )

        # 定期验证
        if episode > 0 and episode % val_interval == 0:
            val_summary = evaluate(val_env, agent, cfg, val_episodes, seed_base=50000)
            val_entry = {"timestamp": timestamp, "episode": episode}
            val_entry.update(val_summary)
            val_log.write(json.dumps(val_entry, ensure_ascii=False) + "\n")
            val_log.flush()

            print(
                f"[Val]   Ep {episode:4d} | Reward: {val_summary['reward_mean']:8.2f} "
                f"± {val_summary['reward_std']:.2f} | "
                f"Completed: {val_summary['completed_mean']} | "
                f"OOM: {val_summary['oom_mean']} | "
                f"SLA Viol: {val_summary['sla_violations_mean']}"
            )

            # 保存最优模型
            if val_summary["reward_mean"] > best_val_reward:
                best_val_reward = val_summary["reward_mean"]
                agent.save(os.path.join(checkpoint_dir, f"a2c_best_{timestamp}.pt"))
                print(f"  -> 新最优模型已保存 (val_reward={best_val_reward:.2f})")

        # 定期保存检查点
        if episode > 0 and episode % save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"a2c_ep{episode}_{timestamp}.pt"))

    # 训练结束，保存最终模型
    agent.save(os.path.join(checkpoint_dir, f"a2c_final_{timestamp}.pt"))

    # 加载最优模型进行测试
    best_model_path = os.path.join(checkpoint_dir, f"a2c_best_{timestamp}.pt")
    if os.path.exists(best_model_path):
        agent.load(best_model_path)
        print(f"\n已加载最优模型: {best_model_path}")

    test_cfg = train_cfg.get("test_episodes", 10)
    test_summary = evaluate(test_env, agent, cfg, test_cfg, seed_base=60000)

    test_log_path = os.path.join(log_dir, f"test_{timestamp}.jsonl")
    with open(test_log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": timestamp, **test_summary}, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("测试集评估结果（最优模型）")
    print("=" * 60)
    for k, v in test_summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    train_log.close()
    val_log.close()
    print(f"\n训练完成。日志保存在: {log_dir}")


if __name__ == "__main__":
    train()

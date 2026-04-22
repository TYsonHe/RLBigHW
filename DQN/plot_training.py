# DQN/plot_training.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（若系统无 SimHei，会自动回退）
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 输出目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "logs", "training_log.jsonl")
SAVE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)


def load_logs(path: str):
    """读取 jsonl 训练日志"""
    records = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到日志文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_reward(records, save_dir):
    """图1: Episode Reward（原始值，无平滑）"""
    episodes = [r["episode"] for r in records]
    rewards = [r["reward"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, color="steelblue", linewidth=1.0, alpha=0.9, label="Raw Reward")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("DQN Training Reward Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_curve.png"), dpi=300)
    plt.close()
    print("[Saved] reward_curve.png")


def plot_loss(records, save_dir):
    """图2: Loss 曲线"""
    episodes = []
    losses = []
    for r in records:
        if r.get("loss") is not None:
            episodes.append(r["episode"])
            losses.append(r["loss"])

    if not losses:
        print("[Skip] No loss data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, losses, color="darkorange", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Huber Loss", fontsize=12)
    ax.set_title("DQN Training Loss Curve", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()
    print("[Saved] loss_curve.png")


def plot_epsilon(records, save_dir):
    """图3: Epsilon 衰减曲线"""
    episodes = [r["episode"] for r in records]
    epsilons = [r["epsilon"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, epsilons, color="green", linewidth=2)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Epsilon", fontsize=12)
    ax.set_title("Exploration Rate (ε-Greedy) Decay", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "epsilon_decay.png"), dpi=300)
    plt.close()
    print("[Saved] epsilon_decay.png")


def plot_throughput(records, save_dir):
    """图4: 每轮完成的请求数（原始值，无平滑）"""
    episodes = [r["episode"] for r in records]
    completed = [r["completed"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, completed, color="teal", linewidth=1.0, alpha=0.9, label="Raw Throughput")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Completed Requests", fontsize=12)
    ax.set_title("System Throughput per Episode", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "throughput_curve.png"), dpi=300)
    plt.close()
    print("[Saved] throughput_curve.png")


def plot_violations(records, save_dir):
    """图5: OOM 与 SLA 违规统计"""
    episodes = [r["episode"] for r in records]
    ooms = [r["oom"] for r in records]
    sla_vios = [r["sla_violations"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, ooms, color="red", linewidth=1.5, label="OOM Count", alpha=0.8)
    ax.plot(episodes, sla_vios, color="purple", linewidth=1.5, label="SLA Violations", alpha=0.8)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Violation Count", fontsize=12)
    ax.set_title("OOM & SLA Violations over Training", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "violations_curve.png"), dpi=300)
    plt.close()
    print("[Saved] violations_curve.png")


def plot_dashboard(records, save_dir):
    """图6: 汇总仪表盘 (2x3 子图，全部原始值)"""
    episodes = [r["episode"] for r in records]
    rewards = [r["reward"] for r in records]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("DQN Training Dashboard", fontsize=16, fontweight="bold", y=1.02)

    # 1. Reward（原始）
    ax = axes[0, 0]
    ax.plot(episodes, rewards, color="steelblue", linewidth=1.0, alpha=0.9)
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    # 2. Loss
    ax = axes[0, 1]
    loss_eps = [r["episode"] for r in records if r.get("loss") is not None]
    loss_vals = [r["loss"] for r in records if r.get("loss") is not None]
    ax.plot(loss_eps, loss_vals, color="darkorange", lw=1.2)
    ax.set_title("Loss (log scale)")
    ax.set_xlabel("Episode")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    # 3. Epsilon
    ax = axes[0, 2]
    ax.plot(episodes, [r["epsilon"] for r in records], color="green", lw=2)
    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    # 4. Throughput（原始）
    ax = axes[1, 0]
    completed = [r["completed"] for r in records]
    ax.plot(episodes, completed, color="teal", linewidth=1.0, alpha=0.9)
    ax.set_title("Completed Requests")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    # 5. OOM
    ax = axes[1, 1]
    ax.plot(episodes, [r["oom"] for r in records], color="red", lw=1.5)
    ax.set_title("OOM Count")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    # 6. SLA Violations
    ax = axes[1, 2]
    ax.plot(episodes, [r["sla_violations"] for r in records], color="purple", lw=1.5)
    ax.set_title("SLA Violations")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_dashboard.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[Saved] training_dashboard.png")


def print_summary(records):
    """打印数值摘要"""
    rewards = [r["reward"] for r in records]
    losses = [r["loss"] for r in records if r.get("loss") is not None]
    completed = [r["completed"] for r in records]
    ooms = [r["oom"] for r in records]
    sla_vios = [r["sla_violations"] for r in records]

    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"Total Episodes      : {len(records)}")
    print(f"Final Epsilon       : {records[-1]['epsilon']:.4f}")
    print(f"Avg Episode Reward  : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Max Episode Reward  : {np.max(rewards):.2f}")
    print(f"Min Episode Reward  : {np.min(rewards):.2f}")
    print(f"Last 100 Ep Avg Rew : {np.mean(rewards[-100:]):.2f}")
    if losses:
        print(f"Final Loss          : {losses[-1]:.6f}")
    print(f"Total Completed     : {sum(completed)}")
    print(f"Total OOM           : {sum(ooms)}")
    print(f"Total SLA Violations: {sum(sla_vios)}")
    print("=" * 50 + "\n")


def main():
    print(f"Loading logs from: {LOG_FILE}")
    records = load_logs(LOG_FILE)
    print(f"Loaded {len(records)} records.")

    print("Generating plots...")
    plot_reward(records, SAVE_DIR)
    plot_loss(records, SAVE_DIR)
    plot_epsilon(records, SAVE_DIR)
    plot_throughput(records, SAVE_DIR)
    plot_violations(records, SAVE_DIR)
    plot_dashboard(records, SAVE_DIR)

    print_summary(records)
    print(f"All figures saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
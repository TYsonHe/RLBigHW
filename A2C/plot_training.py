# A2C/plot_training.py
"""A2C 训练日志可视化。支持指定时间戳或自动选取最新日志。"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAVE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)


def find_latest_log(prefix: str, timestamp: str = None) -> str:
    """查找指定前缀的最新日志文件，或按时间戳精确匹配。"""
    if timestamp:
        path = os.path.join(LOG_DIR, f"{prefix}_{timestamp}.jsonl")
        if os.path.exists(path):
            return path
        print(f"[WARN] 未找到 {path}，回退到最新文件")

    pattern = os.path.join(LOG_DIR, f"{prefix}_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到 {prefix} 日志: {pattern}")
    return files[-1]


def load_logs(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def smooth(values: list, window: int = 20) -> list:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid").tolist()


def plot_reward(records, save_dir, prefix=""):
    episodes = [r["episode"] for r in records]
    rewards = [r["reward"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, color="steelblue", linewidth=0.8, alpha=0.5, label="Raw")
    if len(rewards) > 20:
        s = smooth(rewards)
        ax.plot(episodes[len(episodes)-len(s):], s, color="navy",
                linewidth=2, label=f"MA(20)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(f"{prefix}A2C Training Reward Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "reward_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_losses(records, save_dir, prefix=""):
    episodes = [r["episode"] for r in records]
    policy_losses = [r.get("policy_loss") for r in records]
    value_losses = [r.get("value_loss") for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Policy Loss
    ep_pl = [(e, v) for e, v in zip(episodes, policy_losses) if v is not None]
    if ep_pl:
        ax = axes[0]
        ax.plot([e for e, _ in ep_pl], [v for _, v in ep_pl],
                color="darkorange", linewidth=1.0, alpha=0.7)
        vals = [v for _, v in ep_pl]
        if len(vals) > 20:
            s = smooth(vals)
            ax.plot([e for e, _ in ep_pl][len(ep_pl)-len(s):], s,
                    color="red", linewidth=2, label="MA(20)")
        ax.set_title("Policy Loss")
        ax.set_xlabel("Episode")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Value Loss
    ep_vl = [(e, v) for e, v in zip(episodes, value_losses) if v is not None]
    if ep_vl:
        ax = axes[1]
        ax.plot([e for e, _ in ep_vl], [v for _, v in ep_vl],
                color="teal", linewidth=1.0, alpha=0.7)
        vals = [v for _, v in ep_vl]
        if len(vals) > 20:
            s = smooth(vals)
            ax.plot([e for e, _ in ep_vl][len(ep_vl)-len(s):], s,
                    color="darkblue", linewidth=2, label="MA(20)")
        ax.set_title("Value Loss")
        ax.set_xlabel("Episode")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{prefix}A2C Loss Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_entropy(records, save_dir, prefix=""):
    episodes = [r["episode"] for r in records]
    entropies = [r.get("entropy") for r in records]
    ep_ent = [(e, v) for e, v in zip(episodes, entropies) if v is not None]

    if not ep_ent:
        print("[Skip] No entropy data.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([e for e, _ in ep_ent], [v for _, v in ep_ent],
            color="green", linewidth=1.0, alpha=0.7, label="Raw")
    vals = [v for _, v in ep_ent]
    if len(vals) > 20:
        s = smooth(vals)
        ax.plot([e for e, _ in ep_ent][len(ep_ent)-len(s):], s,
                color="darkgreen", linewidth=2, label="MA(20)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Entropy", fontsize=12)
    ax.set_title(f"{prefix}Policy Entropy (Exploration)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "entropy_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_throughput(records, save_dir, prefix=""):
    episodes = [r["episode"] for r in records]
    completed = [r["completed"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, completed, color="teal", linewidth=0.8, alpha=0.5, label="Raw")
    if len(completed) > 20:
        s = smooth(completed)
        ax.plot(episodes[len(episodes)-len(s):], s, color="darkcyan",
                linewidth=2, label="MA(20)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Completed Requests", fontsize=12)
    ax.set_title(f"{prefix}System Throughput per Episode", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "throughput_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_violations(records, save_dir, prefix=""):
    episodes = [r["episode"] for r in records]
    ooms = [r["oom"] for r in records]
    sla_vios = [r["sla_violations"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, ooms, color="red", linewidth=1.0, alpha=0.6, label="OOM")
    ax.plot(episodes, sla_vios, color="purple", linewidth=1.0, alpha=0.6, label="SLA Violations")

    if len(ooms) > 20:
        s_oom = smooth(ooms)
        ax.plot(episodes[len(episodes)-len(s_oom):], s_oom,
                color="darkred", linewidth=2, label="OOM MA(20)")
    if len(sla_vios) > 20:
        s_sla = smooth(sla_vios)
        ax.plot(episodes[len(episodes)-len(s_sla):], s_sla,
                color="indigo", linewidth=2, label="SLA MA(20)")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Violation Count", fontsize=12)
    ax.set_title(f"{prefix}OOM & SLA Violations", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "violations_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_cost(records, save_dir, prefix=""):
    episodes = [r["episode"] for r in records]
    costs = [r.get("total_cost", 0) for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, costs, color="orange", linewidth=0.8, alpha=0.5, label="Raw")
    if len(costs) > 20:
        s = smooth(costs)
        ax.plot(episodes[len(episodes)-len(s):], s, color="brown",
                linewidth=2, label="MA(20)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Cost", fontsize=12)
    ax.set_title(f"{prefix}Operational Cost per Episode", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "cost_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_val_curve(val_records, save_dir):
    """绘制验证集指标随训练进度变化。"""
    if not val_records:
        print("[Skip] 无验证日志。")
        return

    episodes = [r["episode"] for r in val_records]
    reward_mean = [r["reward_mean"] for r in val_records]
    reward_std = [r["reward_std"] for r in val_records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, reward_mean, color="royalblue", linewidth=2, marker="o",
            markersize=4, label="Val Reward (mean)")
    ax.fill_between(episodes,
                    [m - s for m, s in zip(reward_mean, reward_std)],
                    [m + s for m, s in zip(reward_mean, reward_std)],
                    color="royalblue", alpha=0.15)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Validation Reward", fontsize=12)
    ax.set_title("A2C Validation Reward Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "val_reward_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] {path}")

    # 验证集多指标子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("A2C Validation Metrics", fontsize=14, fontweight="bold")

    metrics = [
        ("completed_mean", "Avg Completed", "teal"),
        ("oom_mean", "Avg OOM", "red"),
        ("sla_violations_mean", "Avg SLA Violations", "purple"),
        ("total_cost_mean", "Avg Cost", "orange"),
    ]
    for ax, (key, title, color) in zip(axes.flat, metrics):
        vals = [r.get(key, 0) for r in val_records]
        ax.plot(episodes, vals, color=color, linewidth=2, marker="o", markersize=4)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "val_metrics.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] {path}")


def plot_dashboard(records, val_records, save_dir):
    """汇总仪表盘：2×3 子图。"""
    episodes = [r["episode"] for r in records]
    rewards = [r["reward"] for r in records]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("A2C Training Dashboard", fontsize=16, fontweight="bold")

    # 1. Reward
    ax = axes[0, 0]
    ax.plot(episodes, rewards, color="steelblue", lw=0.8, alpha=0.5)
    if len(rewards) > 20:
        s = smooth(rewards)
        ax.plot(episodes[len(episodes)-len(s):], s, color="navy", lw=2, label="MA(20)")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    # 2. Policy Loss
    ax = axes[0, 1]
    pl = [(e, r["policy_loss"]) for e, r in zip(episodes, records) if r.get("policy_loss") is not None]
    if pl:
        ax.plot([e for e, _ in pl], [v for _, v in pl], color="darkorange", lw=1)
    ax.set_title("Policy Loss")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    # 3. Entropy
    ax = axes[0, 2]
    ent = [(e, r["entropy"]) for e, r in zip(episodes, records) if r.get("entropy") is not None]
    if ent:
        ax.plot([e for e, _ in ent], [v for _, v in ent], color="green", lw=1)
    ax.set_title("Policy Entropy")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    # 4. Throughput
    ax = axes[1, 0]
    ax.plot(episodes, [r["completed"] for r in records], color="teal", lw=0.8, alpha=0.5)
    ax.set_title("Completed Requests")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    # 5. OOM
    ax = axes[1, 1]
    ax.plot(episodes, [r["oom"] for r in records], color="red", lw=1)
    ax.set_title("OOM Count")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    # 6. SLA Violations
    ax = axes[1, 2]
    ax.plot(episodes, [r["sla_violations"] for r in records], color="purple", lw=1)
    ax.set_title("SLA Violations")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_dashboard.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def print_summary(records, val_records, test_records):
    print("\n" + "=" * 60)
    print("A2C Training Summary")
    print("=" * 60)

    if records:
        rewards = [r["reward"] for r in records]
        print(f"训练集 Episodes       : {len(records)}")
        print(f"平均 Episode Reward   : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"最大 Episode Reward   : {np.max(rewards):.2f}")
        print(f"最小 Episode Reward   : {np.min(rewards):.2f}")
        print(f"最后 100 Ep 平均 Reward : {np.mean(rewards[-100:]):.2f}")

        total_completed = sum(r["completed"] for r in records)
        total_oom = sum(r["oom"] for r in records)
        total_sla = sum(r["sla_violations"] for r in records)
        print(f"总完成请求数          : {total_completed}")
        print(f"总 OOM 次数           : {total_oom}")
        print(f"总 SLA 违规次数       : {total_sla}")

    if val_records:
        print(f"\n验证集评估次数        : {len(val_records)}")
        last_val = val_records[-1]
        print(f"最终验证 Reward       : {last_val['reward_mean']:.2f} ± {last_val['reward_std']:.2f}")
        print(f"最终验证 OOM          : {last_val['oom_mean']}")
        print(f"最终验证 SLA 违规     : {last_val['sla_violations_mean']}")

    if test_records:
        test = test_records[0]
        print(f"\n测试集评估结果        :")
        for k, v in test.items():
            print(f"  {k}: {v}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="A2C 训练日志可视化")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="指定训练时间戳（如 20240422_153000），不指定则使用最新日志")
    args = parser.parse_args()

    ts = args.timestamp
    train_path = find_latest_log("train", ts)
    val_path = find_latest_log("val", ts)

    # 测试日志可选
    test_records = []
    try:
        test_path = find_latest_log("test", ts)
        test_records = load_logs(test_path)
        print(f"加载测试日志: {test_path} ({len(test_records)} 条)")
    except FileNotFoundError:
        print("[INFO] 未找到测试日志，跳过")

    print(f"加载训练日志: {train_path}")
    records = load_logs(train_path)
    print(f"加载验证日志: {val_path}")
    val_records = load_logs(val_path)

    print(f"训练记录: {len(records)} 条 | 验证记录: {len(val_records)} 条")
    print("生成图表...")

    plot_reward(records, SAVE_DIR)
    plot_losses(records, SAVE_DIR)
    plot_entropy(records, SAVE_DIR)
    plot_throughput(records, SAVE_DIR)
    plot_violations(records, SAVE_DIR)
    plot_cost(records, SAVE_DIR)
    plot_val_curve(val_records, SAVE_DIR)
    plot_dashboard(records, val_records, SAVE_DIR)

    print_summary(records, val_records, test_records)
    print(f"所有图表保存在: {SAVE_DIR}")


if __name__ == "__main__":
    main()

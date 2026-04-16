"""
环境演示脚本：展示 LLMClusterEnv 的基本使用
"""
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.envs.cluster_env import LLMClusterEnv


def demo_random_episode():
    """演示随机策略运行一个 episode"""
    print("=" * 60)
    print("演示: LLMClusterEnv 随机策略")
    print("=" * 60)

    # 创建环境
    env = LLMClusterEnv(num_nodes=10)

    # 重置环境
    obs, info = env.reset(seed=42)
    print(f"\n初始观测值形状: {obs.shape}")
    print(f"当前工作负载: {info['workload_queue']}")

    # 运行一个 episode
    total_reward = 0
    num_steps = 100

    print(f"\n运行 {num_steps} 步...")
    for step in range(num_steps):
        # 随机选择动作
        action = env.action_space.sample()

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 打印关键步骤
        if step % 20 == 0:
            print(f"  Step {step:3d}: action={action}, reward={reward:+.2f}, "
                  f"completed={info['completed_requests']}")

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    print(f"\n总奖励: {total_reward:.2f}")
    print(f"平均奖励: {total_reward / num_steps:.2f}")

    env.close()
    print("\n演示完成!")


if __name__ == "__main__":
    demo_random_episode()
# Project.md — RLBigHW 项目完整分析

## 1. 项目概述

### 1.1 研究问题

大型语言模型（LLM）推理服务在异构 GPU 集群上的部署面临核心挑战：**如何在满足不同优先级请求的 SLA 延迟要求的同时，最大化集群资源利用率并降低运营成本？** 传统的静态资源分配方案无法应对请求流量和负载特征的动态变化，而强化学习（Reinforcement Learning, RL）为这一在线决策问题提供了自然的建模框架。

### 1.2 核心思路

将异构 GPU 集群的节点划分为 **3 个资源池**（短请求池 / 长请求池 / 混合池），由 RL Agent 在每个时间步根据集群当前状态重新配置池参数（节点分配比例、并发实例数、流量路由权重、显存限制、卸载策略等），以多目标加权奖励信号引导学习，实现吞吐量、延迟、利用率、成本与 SLA 的联合优化。

### 1.3 已实现算法

| 算法 | 类型 | 核心机制 | 状态 |
|------|------|----------|------|
| **Dueling Double DQN** | 值函数方法 | 经验回放 + 目标网络 + Dueling 架构 | 已完成 |
| **A2C (Advantage Actor-Critic)** | 策略梯度方法 | GAE 优势估计 + 熵正则 + 共享骨干 | 已完成 |

两种算法共享相同的 RL 环境（`LLMClusterEnv`），具有相同的状态空间（14 维）和动作空间（36 离散），可直接对比性能。

---

## 2. 仓库结构

```
RLBigHW/
├── data/                                    # 仿真数据集
│   ├── cluster_profiles_train.csv           # 集群节点配置（500 节点）
│   ├── cluster_profiles_val.csv             # 验证集（100 节点）
│   ├── cluster_profiles_test.csv            # 测试集（50 节点）
│   ├── workload_streams_train.csv           # 工作负载流（~50000 请求，7 天）
│   ├── workload_streams_val.csv             # 验证负载（~10000 请求，1 天）
│   └── workload_streams_test.csv            # 测试负载（~10000 请求，1 天）
├── src/
│   ├── envs/
│   │   └── cluster_env.py                   # 核心 RL 环境（Gymnasium 兼容）
│   ├── utils/data_generators/
│   │   ├── cluster_generator.py             # 集群拓扑生成器
│   │   └── workload_generator.py            # 工作负载流生成器
│   └── generate_datasets.py                 # 数据集生成入口脚本
├── DQN/                                     # DQN 算法目录
│   ├── algorithms/
│   │   ├── dqn.py                           # Dueling DQN 网络 + Agent
│   │   └── replay_buffer.py                 # 经验回放缓冲区
│   ├── train_dqn.py                         # 训练脚本
│   ├── plot_training.py                     # 训练可视化（6 图 + 仪表盘）
│   ├── checkpoints/                         # 模型检查点
│   ├── logs/training_log.jsonl              # 训练日志
│   └── figures/                             # 训练曲线图
├── A2C/                                     # A2C 算法目录
│   ├── algorithms/
│   │   ├── __init__.py                      # 导出 A2CAgent, ActorCritic, RolloutBuffer
│   │   ├── a2c.py                           # ActorCritic 网络 + A2CAgent
│   │   └── rollout_buffer.py                # Rollout 缓冲区（支持 GAE）
│   ├── configs/
│   │   └── default.yaml                     # 超参数配置文件
│   ├── train_a2c.py                         # 训练脚本（训练/验证/测试 + GPU）
│   ├── plot_training.py                     # 训练可视化（9 图 + 仪表盘）
│   ├── checkpoints/                         # 模型检查点
│   ├── logs/                                # 时间戳日志（JSONL + 配置快照）
│   └── figures/                             # 训练曲线图
├── requirements.txt                         # 依赖清单
├── Project.md                               # 本文档
└── CLAUDE.md                                # 开发指南
```

---

## 3. RL 环境详细分析 — `LLMClusterEnv`

### 3.1 环境概述

`LLMClusterEnv` 是基于 Gymnasium 框架的定制 RL 环境，模拟异构 GPU 集群上 LLM 推理服务的资源调度。环境将集群节点划分为 3 个资源池，每步由 Agent 选择池配置参数，环境模拟请求到达、准入、处理和奖励计算。

**环境 ID**：`LLMClusterEnv`（非 Gymnasium 注册，需直接实例化）

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_nodes` | 100 | 集群节点总数 |
| `step_batch` | 10 | 每步到达的请求数量 |
| `reward_weights` | {tp:0.3, lat:0.3, util:0.2, cost:0.1, sla:0.1} | 奖励各分量权重 |

### 3.2 状态空间（14 维连续向量）

| 维度索引 | 含义 | 计算方式 | 典型范围 |
|----------|------|----------|----------|
| 0 | 短请求池最大队列长度 | `max(queues) / 100.0` | [0, +∞) |
| 1 | 短请求池平均 GPU 显存利用率 | `mean(gpu_mem_util)` | [0, 1] |
| 2 | 短请求池活跃节点比例 | `len(active) / num_nodes_total` | [0, 1] |
| 3 | 长请求池最大队列长度 | 同维度 0 | [0, +∞) |
| 4 | 长请求池平均 GPU 显存利用率 | 同维度 1 | [0, 1] |
| 5 | 长请求池活跃节点比例 | 同维度 2 | [0, 1] |
| 6 | 混合池最大队列长度 | 同维度 0 | [0, +∞) |
| 7 | 混合池平均 GPU 显存利用率 | 同维度 1 | [0, 1] |
| 8 | 混合池活跃节点比例 | 同维度 2 | [0, 1] |
| 9 | 等待队列 P95 等待时间 | `percentile(wait_times, 95) / 10.0` | [0, +∞) |
| 10 | 长请求占比 | `count(len>8000) / total` | [0, 1] |
| 11 | 高优先级请求占比 | `count(priority==1) / total` | [0, 1] |
| 12 | 时间特征（小时正弦） | `sin(2π × hour / 24)` | [-1, 1] |
| 13 | 时间特征（小时余弦） | `cos(2π × hour / 24)` | [-1, 1] |

**设计要点**：
- **每池 3 维**（队列深度 / 显存利用率 / 节点占比），3 池共 9 维——提供池级细粒度感知
- **3 维全局负载特征**（等待时间 / 长请求比 / 高优先级比）——捕捉请求结构信息
- **2 维周期时间编码**——正弦/余弦编码保证 23:59 与 00:00 在向量空间中距离最小，避免边界不连续

### 3.3 动作空间（36 个离散动作）

动作编码为 `action_id ∈ [0, 35]`，通过三重分解映射到池配置：

```
action_id = a × 12 + b × 4 + c
```

| 因子 | 范围 | 含义 |
|------|------|------|
| `a = action_id // 12` | {0, 1, 2} | **流量分配模式**（gamma 分布），决定请求路由概率 |
| `b = (action_id % 12) // 4` | {0, 1, 2} | **并发度偏移**（beta 调整量），相对基础值 -1/0/+2 |
| `c = action_id % 4` | {0, 1, 2, 3} | **资源限制预设**（v/m 组合），控制显存上限与卸载策略 |

**三种流量分配模式** (`a`)：

| a | 短池 γ | 长池 γ | 混合池 γ | 调度语义 |
|---|--------|--------|----------|----------|
| 0 | 0.7 | 0.2 | 0.1 | 偏向短请求——适用于轻量推理高峰 |
| 1 | 0.2 | 0.7 | 0.1 | 偏向长请求——适用于长文本生成场景 |
| 2 | 0.4 | 0.4 | 0.2 | 均衡分配——通用场景 |

**四种资源限制预设** (`c`)：

| c | v（显存上限） | m（卸载阈值） | 资源策略 |
|---|--------------|--------------|----------|
| 0 | 0.4 | 0.2 | **保守**：低显存占用，易触发 CPU 卸载，稳定性优先 |
| 1 | 0.6 | 0.4 | **适中**：平衡吞吐与稳定性 |
| 2 | 0.8 | 0.6 | **激进**：高显存占用，减少卸载，吞吐优先 |
| 3 | 1.0 | 0.9 | **极限**：几乎不卸载，最大化 GPU 利用 |

> 注意：短请求池的 v 上限被额外裁剪为 `min(v, 0.8)`，防止短请求过度占用显存。

### 3.4 池配置参数体系

每个池（short=0, long=1, mixed=2）有 6 个参数，Agent 通过动作间接控制这些参数：

| 参数 | 符号 | 含义 | 初始值（短/长/混合） | Agent 影响方式 |
|------|------|------|---------------------|---------------|
| 节点比例 | `alpha` | 分配到该池的节点占比 | 0.4 / 0.4 / 0.2 | 固定（环境默认） |
| 并发实例数 | `beta` | 池内总推理实例数（均摊到节点） | 2 / 1 / 1 | 由因子 `b` 调整偏移量 |
| 流量比例 | `gamma` | 请求路由到该池的概率权重 | 0.7 / 0.3 / 0.0 | 由因子 `a` 选择模式 |
| 显存上限 | `v` | 单节点可用显存占总量的比例 | 0.6 / 1.0 / 0.8 | 由因子 `c` 选择预设 |
| 卸载阈值 | `m` | 超过此值触发 CPU offload | 0.2 / 0.8 / 0.5 | 由因子 `c` 选择预设 |
| 计算缩放 | `c` | 推理速度乘数 | 0.5 / 0.9 / 0.6 | 由因子 `c` 间接计算：`0.5 + 0.4 × (c/3)` |

### 3.5 奖励函数

多目标加权奖励，5 项指标加权求和 + 惩罚项：

```
R = w_tp × R_throughput + w_lat × R_latency + w_util × R_utilization
  + w_cost × R_cost + w_sla × R_sla + penalty
```

| 分量 | 公式 | 默认权重 | 语义 |
|------|------|----------|------|
| `R_throughput` | `completed / 10.0` | 0.3 | 吞吐量奖励：完成请求越多奖励越大 |
| `R_latency` | `-mean(latency) / 500.0` | 0.3 | 延迟惩罚：平均延迟越高惩罚越大 |
| `R_utilization` | `-mean(|util - 0.7|)` | 0.2 | 利用率偏离惩罚：鼓励 70% 最佳利用率 |
| `R_cost` | `-cost / 10.0` | 0.1 | 运营成本惩罚 |
| `R_sla` | `-sla_violations` | 0.1 | SLA 违规计数惩罚（延迟 > 200ms 为违规） |

**惩罚项**：
- 请求被拒但可卸载（`m > 0.5`）：`-5.0`
- 请求被拒且不可卸载（`m ≤ 0.5`，即 OOM）：`-10.0`，同时 `stats["oom"] += 1`

**奖励尺度分析**：
- 正常单步 reward 范围：约 [-500, +50]（高 OOM 时可达 -25000+）
- A2C 训练中对 reward 裁剪为 [-50, 50]，防止惩罚爆炸导致价值函数发散

### 3.6 环境核心机制

#### 节点初始化与池分配
- 从 CSV 读取集群配置，按 GPU 型号归类（A100 / L40S / 4090）
- 池分配偏好：短请求池偏好 4090（低成本推理），长请求池偏好 A100（高算力），混合池无偏好
- 节点按 `cost_per_hour` 升序排序后依次分配到各池
- 每个节点的实例数 = `beta // pool_size + (1 if idx < remainder else 0)`，并受 `v` 限制上限

#### 请求到达
- 每个 `step` 从工作负载 CSV 批量读取 `step_batch=10` 条请求
- 请求按 `gamma` 分布的概率路由到对应池（`np.random.choice(3, p=[γ_0, γ_1, γ_2])`）
- 最小队列节点优先接收请求

#### 请求准入与卸载
- 计算请求 VRAM 需求：`weight_gb + kv_per_token_kb × total_tokens / 1024`
- 检查节点显存余量：`当前已用 + VRAM需求 ≤ v × gpu_mem_total_gb`？
  - 若满足：准入，更新显存利用率
  - 若不满足，检查卸载阈值 `m`：
    - `m > 0.5`：允许 CPU 卸载，延迟乘以 `1.5 + (1.0 - m)`
    - `m ≤ 0.5`：拒绝请求，返回惩罚

#### 请求处理
- 处理速率 = `1000 / ms_per_token × instances × c`（tokens/sec）
- 处理数 = `min(queue, max(1, proc_rate / 500))`
- GPU 显存利用率每步衰减 0.8 倍（模拟请求完成释放显存）
- 延迟基准：`uniform(50, 500)`，卸载时 ×1.5
- SLA 阈值：200ms（延迟超过 200ms 计为 SLA 违规）

#### 终止条件
- `terminated`：工作负载缓冲区为空 且 已运行 >1000 步
- `truncated`：超过 5000 步强制截断

### 3.7 LLM 模型规格

| 模型 | 权重大小 | KV/Token | A100 延迟 | L40S 延迟 | 4090 延迟 |
|------|---------|----------|----------|----------|----------|
| LLaMA3-7B | 14 GB | 2 KB | 2.5 ms | 3.2 ms | 4.0 ms |
| Qwen-14B | 28 GB | 4 KB | 4.0 ms | 5.5 ms | 8.0 ms |
| DeepSeek-70B | 140 GB | 16 KB | 12.0 ms | 18.0 ms | ∞（显存不足） |

---

## 4. Dueling Double DQN 算法

### 4.1 算法原理

Dueling Double DQN 是值函数方法的代表，通过三个关键技术改进标准 DQN：

1. **Double DQN**：将动作选择与 Q 值评估解耦，用 `policy_net` 选最优动作，`target_net` 评估该动作的 Q 值，避免标准 DQN 的 Q 值过估计问题
2. **Dueling 架构**：将 Q 函数分解为状态价值 V(s) 和动作优势 A(s,a)，使网络更高效地学习哪些状态本身有价值
3. **经验回放**：打破样本间的时间相关性，提高数据利用效率

### 4.2 网络结构

```
输入 (state_dim=14)
    │
    ▼
┌──────────────────────────────────────────┐
│  共享骨干 (Backbone)                      │
│  Linear(14, 256) → LayerNorm → ReLU → Dropout(0.1)  │
│  Linear(256, 256) → LayerNorm → ReLU → Dropout(0.1) │
└──────────────┬───────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐ ┌──────────────────┐
│  Value 流     │ │  Advantage 流     │
│  Linear(256,128)│ │  Linear(256,128)  │
│  ReLU           │ │  ReLU             │
│  Linear(128,1)  │ │  Linear(128,36)   │
└──────┬───────┘ └────────┬──────────┘
       │                  │
       ▼                  ▼
     V(s)             A(s,a)
       │                  │
       └──────┬───────────┘
              ▼
  Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
```

- **参数量**：~110K（骨干 + Value 流 + Advantage 流）
- **正则化**：LayerNorm 稳定训练 + Dropout(0.1) 防过拟合
- **Dueling 合并公式**：`Q(s,a) = V(s) + A(s,a) - mean(A(s,:))`，确保 V(s) 和 A(s,a) 的可辨识性

### 4.3 训练机制

| 机制 | 实现 | 作用 |
|------|------|------|
| **Double DQN** | `a* = argmax_a Q_policy(s',a)`，`y = r + γ × Q_target(s', a*)` | 避免过估计 |
| **经验回放** | 容量 100K，均匀采样 batch=256 | 打破时间相关性 |
| **目标网络** | 每 10 步 soft update (τ=0.005)，每 50 episode 硬同步 | 稳定训练目标 |
| **损失函数** | Smooth L1 Loss（Huber Loss） | 对异常值鲁棒 |
| **梯度裁剪** | `max_norm=10.0` | 防梯度爆炸 |
| **探索策略** | ε-greedy，指数衰减 `ε = 0.01 + 0.99 × exp(-step/20000)` | 从探索到利用的平滑过渡 |

### 4.4 超参数汇总

| 参数 | 值 | 说明 |
|------|-----|------|
| `state_dim` | 14 | 状态向量维度 |
| `action_dim` | 36 | 离散动作数 |
| `lr` | 1e-4 | Adam 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `epsilon_start` | 1.0 | 初始探索率 |
| `epsilon_end` | 0.01 | 最终探索率 |
| `epsilon_decay` | 20000 | 衰减步数常数 |
| `buffer_size` | 100000 | 经验回放容量 |
| `batch_size` | 256 | 训练批次大小 |
| `tau` | 0.005 | 软更新系数 |
| `num_episodes` | 2000 | 总训练回合 |
| `max_steps` | 500 | 每回合最大步数 |
| `hidden_dims` | [256, 256] | 隐藏层维度 |
| `dropout` | 0.1 | Dropout 率 |

---

## 5. A2C (Advantage Actor-Critic) 算法

### 5.1 算法原理

A2C 属于策略梯度方法家族，同时学习策略函数 π(a|s) 和价值函数 V(s)：

1. **Actor（策略网络）**：输出动作概率分布，通过策略梯度更新，目标是最大化 `E[log π(a|s) × A(s,a)]`
2. **Critic（价值网络）**：估计状态价值 V(s)，作为策略梯度的基线（baseline），降低方差
3. **GAE 优势估计**：使用广义优势估计（Generalized Advantage Estimation）在偏差和方差之间取得平衡
4. **熵正则化**：在损失中加入策略熵奖励，鼓励探索，替代 ε-greedy 机制

**与 DQN 的核心区别**：

| 维度 | DQN | A2C |
|------|-----|-----|
| 学习对象 | Q 值函数 Q(s,a) | 策略 π(a\|s) + 价值 V(s) |
| 动作选择 | ε-greedy（Q 值最大 + 随机探索） | 从策略分布采样（探索内生于策略） |
| 更新方式 | 离线回放（off-policy） | 在线 rollout（on-policy） |
| 样本效率 | 高（经验回放复用数据） | 低（每条数据仅用一次） |
| 稳定性 | 依赖目标网络 | 依赖优势估计与梯度裁剪 |
| 探索机制 | ε 衰减 | 熵正则化 |

### 5.2 网络结构 — ActorCritic

```
输入 (state_dim=14)
    │
    ▼
┌──────────────────────────────────────────┐
│  共享骨干 (Backbone)                      │
│  Linear(14, 256) → LayerNorm → ReLU      │
│  Linear(256, 256) → LayerNorm → ReLU     │
└──────────────┬───────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐ ┌──────────────────┐
│  策略头        │ │  价值头           │
│  Linear(256,128)│ │  Linear(256,128)  │
│  ReLU           │ │  ReLU             │
│  Linear(128,36) │ │  Linear(128,1)    │
└──────┬───────┘ └────────┬──────────┘
       │                  │
       ▼                  ▼
   logits (36)         V(s) (1)
       │
  Categorical(logits)
       │
       ▼
  π(a|s) — 动作概率分布
```

- **参数量**：~120K（骨干 + 策略头 + 价值头）
- **共享骨干**：策略和价值共享特征提取器，减少参数量并加速收敛
- **策略头**：输出 36 维 logits，经 `Categorical` 分布采样动作
- **价值头**：输出 1 维标量 V(s)，作为 GAE 计算的基线
- **无 Dropout**：on-policy 算法中 Dropout 可能干扰策略一致性，用 LayerNorm 已足够

### 5.3 GAE 优势估计

GAE（Generalized Advantage Estimation）是 A2C 的核心组件，在偏差和方差之间取得平衡：

**TD 残差**：
```
δ_t = r_t + γ × V(s_{t+1}) × (1 - done_t) - V(s_t)
```

**GAE 递推**：
```
Â_t = δ_t + (γ × λ) × (1 - done_t) × Â_{t+1}
```

**回报**：
```
R_t = Â_t + V(s_t)
```

其中 λ（`gae_lambda`）控制偏差-方差权衡：
- λ = 0：Â_t = δ_t（低方差、高偏差，仅看一步 TD 误差）
- λ = 1：Â_t = Σ γ^k δ_{t+k}（高方差、低偏差，近似蒙特卡洛回报）
- 默认 λ = 0.95：经验上的良好折中

### 5.4 损失函数

A2C 的总损失由三部分组成：

```
L_total = L_policy + α_value × L_value + α_entropy × L_entropy
```

| 分量 | 公式 | 作用 |
|------|------|------|
| `L_policy` | `-mean(log π(a\|s) × Â)` | 策略梯度：增大高优势动作的概率 |
| `L_value` | `SmoothL1(V(s), R)` | 价值函数回归：让 V(s) 逼近实际回报 |
| `L_entropy` | `-mean(H(π(·\|s)))` | 熵奖励：鼓励策略保持探索性，防止坍塌 |

- **优势标准化 + 裁剪**：先标准化 `(Â - μ) / (σ + ε)`，再裁剪到 [-10, 10]，防止梯度爆炸
- **Huber Loss**（Smooth L1）：对异常回报值更鲁棒，相比 MSE 不易被极端值主导
- **梯度裁剪**：`max_norm=0.5`，比 DQN 的 10.0 更严格

### 5.5 训练流程

```
run_episode(env, agent, max_steps, n_steps):
│
├── state = env.reset()
│
└── for t in range(max_steps):
    ├── result = agent.select_action(state)
    │   ├── logits, value = network(state)          # 前向传播
    │   ├── dist = Categorical(logits)               # 构建策略分布
    │   ├── action = dist.sample()                   # 采样动作
    │   └── return {action, log_prob, value}
    │
    ├── next_state, reward, done, info = env.step(action)
    ├── reward = clip(reward, -50, 50)               # 奖励裁剪
    ├── buffer.add(state, action, reward, value, log_prob, done)
    │
    └── if (t+1) % n_steps == 0 or done:             # 每 n_steps 更新一次
        ├── next_value = V(s_{t+1})                   # 估计下一状态价值
        ├── advantages, returns = buffer.compute_gae(next_value)
        ├── 标准化 + 裁剪 advantages
        ├── 前向传播获取当前 logits, values
        ├── 计算 policy_loss + value_loss + entropy_loss
        ├── 反向传播 + 梯度裁剪 + 优化器步进
        └── buffer.clear()
```

**关键设计**：
- **n-step 更新**：每 10 步收集一次 rollout 数据后更新，而非每步更新，提高数据效率
- **奖励裁剪 [-50, 50]**：防止 OOM 惩罚（单步可达 -25000）主导训练信号，避免价值函数发散
- **确定性评估**：验证和测试时使用 `argmax(logits)` 选择动作，消除随机性

### 5.6 超参数汇总

| 参数 | 值 | 说明 |
|------|-----|------|
| `state_dim` | 14 | 状态向量维度 |
| `action_dim` | 36 | 离散动作数 |
| `lr` | 3e-4 | Adam 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 平滑参数 |
| `entropy_coef` | 0.05 | 熵正则系数（控制探索强度） |
| `value_coef` | 0.5 | 价值损失权重 |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 |
| `n_steps` | 10 | Rollout 长度 / 更新频率 |
| `num_episodes` | 500 | 总训练回合 |
| `max_steps` | 200 | 每回合最大步数 |
| `val_interval` | 50 | 验证间隔（episode） |
| `val_episodes` | 5 | 验证回合数 |
| `test_episodes` | 10 | 测试回合数 |
| `hidden_dims` | [256, 256] | 隐藏层维度 |
| `device` | auto | 自动选择 CUDA/CPU |

### 5.7 Rollout 缓冲区

`RolloutBuffer` 存储 n-step rollout 的转移数据，并支持 GAE 计算：

| 方法 | 功能 |
|------|------|
| `add(state, action, reward, value, log_prob, done)` | 添加一条转移记录 |
| `compute_gae(next_value, gamma, gae_lambda)` | 计算优势函数和回报 |
| `get_tensors(device)` | 将数据转为 PyTorch 张量并移至指定设备 |
| `clear()` | 清空缓冲区 |

### 5.8 训练稳定性措施

A2C 训练中遇到了奖励崩溃和策略坍塌问题，通过以下措施解决：

| 措施 | 位置 | 说明 |
|------|------|------|
| **奖励裁剪** [-50, 50] | `train_a2c.py` | 阻止 OOM 惩罚主导训练信号 |
| **优势值裁剪** [-10, 10] | `a2c.py` | 防止标准化后的优势值仍过大导致梯度爆炸 |
| **Huber Loss** | `a2c.py` | 替代 MSE，对异常回报更鲁棒 |
| **增大熵系数** (0.01→0.05) | `default.yaml` | 增强探索，防止策略坍塌到确定性动作 |
| **降低学习率** (7e-4→3e-4) | `default.yaml` | 减少参数更新幅度，提升稳定性 |
| **梯度裁剪** (0.5) | `a2c.py` | 比 DQN 的 10.0 更严格，适应 on-policy 更新 |

---

## 6. 数据生成系统

### 6.1 集群生成器 (`cluster_generator.py`)

**GPU 型号及规格**：

| GPU 型号 | 显存 | 时费（元/h） | 算力 (TFLOPS) | 采样概率 |
|----------|------|-------------|--------------|---------|
| NVIDIA A100-SXM4-80GB | 80 GB | 30.0 | 312 | 20% |
| NVIDIA L40S-48GB | 48 GB | 12.0 | 180 | 30% |
| NVIDIA RTX 4090-24GB | 24 GB | 5.0 | 82 | 50% |

**其他节点属性**：
- CPU 核心：32 / 64 / 128，随机选取
- 内存：256 / 512 GB，随机选取
- 磁盘：2000 / 4000 GB NVMe，随机选取
- 区域：北京 / 上海 / 广州，均匀选取
- 带宽：10000~100000 Mbps，均匀随机
- 时费：基础价 ± 10% 随机波动

**数据规模**：

| 数据集 | 节点数 | 文件大小 |
|--------|--------|----------|
| 训练集 | 500 | 46 KB |
| 验证集 | 100 | 9 KB |
| 测试集 | 50 | 5 KB |

### 6.2 工作负载生成器 (`workload_generator.py`)

**请求到达模型**：非齐次泊松过程

```
λ(t) = λ_base × (1 + A × sin(2π × (hour - φ) / T))
```

- 基础到达率 `λ_base = 50` 次/小时
- 振幅 `A = 0.5`（峰谷比约 3:1）
- 峰值偏移 `φ = 9`（上午 9 点高峰，模拟工作时段）
- 周期 `T = 24h`

**LLM 模型分布**：

| 模型 | 权重大小 | KV/Token | 采样概率 | 典型用途 |
|------|---------|----------|---------|----------|
| LLaMA3-7B | 14 GB | 2 KB | 50% | 轻量推理、对话 |
| Qwen-14B | 28 GB | 4 KB | 35% | 中等复杂度生成 |
| DeepSeek-70B | 140 GB | 16 KB | 15% | 长文本生成、复杂推理 |

**Token 长度分布**：对数正态分布

| 参数 | 输入 tokens | 输出 tokens |
|------|-----------|-----------|
| μ | 7.5 | 6.0 |
| σ | 1.2 | 0.8 |
| 范围 | [50, 4096] | [50, 2048] |

**优先级分布**：P1=20%（高优先级）、P2=50%（标准）、P3=30%（低优先级）

**SLA 延迟阈值**（按模型分）：

| 模型 | 基础阈值 | Token 缩放 |
|------|---------|-----------|
| LLaMA3-7B | 200 ms | +0.1 × (input_tokens / 1000) |
| Qwen-14B | 500 ms | +0.1 × (input_tokens / 1000) |
| DeepSeek-70B | 2000 ms | +0.1 × (input_tokens / 1000) |

**数据规模**：

| 数据集 | 请求数 | 时间跨度 | 文件大小 |
|--------|--------|---------|----------|
| 训练集 | ~50000 | 7 天 | 2.4 MB |
| 验证集 | ~10000 | 1 天 | 475 KB |
| 测试集 | ~10000 | 1 天 | 475 KB |

---

## 7. 训练与评估系统

### 7.1 DQN 训练流程

```
train_dqn.py 主循环：
│
├── 初始化环境 LLMClusterEnv（加载 train 数据）
├── 初始化 DQNAgent（state_dim=14, action_dim=36）
├── 初始化 ReplayBuffer（容量 100K）
│
└── for episode in range(2000):
    ├── state = env.reset()
    └── for t in range(500):
        ├── action = agent.select_action(state)         # ε-greedy 策略
        ├── next_state, reward, done, info = env.step(action)
        ├── replay_buffer.add(state, action, reward, next_state, done)
        ├── agent.learn(replay_buffer)                  # Double DQN 更新
        ├── agent.soft_update_target(τ=0.005)           # 每 10 步软更新
        └── break if done

    ├── 每 50 episode：target_net 硬同步
    ├── 每 100 episode：保存 checkpoint
    └── 写入 training_log.jsonl
```

### 7.2 A2C 训练流程

```
train_a2c.py 主循环：
│
├── 加载配置 configs/default.yaml
├── 设备检测：auto → CUDA/CPU，打印 GPU 信息
├── 固定随机种子（NumPy + PyTorch + CUDA）
├── 保存配置快照 config_{timestamp}.yaml
│
├── 创建 3 个环境：
│   ├── train_env（训练集集群 + 负载）
│   ├── val_env（验证集集群 + 负载）
│   └── test_env（测试集集群 + 负载）
│
├── 创建 A2CAgent（从配置读取超参数）
│
├── 打开日志文件：train_{ts}.jsonl / val_{ts}.jsonl
│
└── for episode in range(500):
    ├── result = run_episode(train_env, agent, 200步, n_steps=10)
    │   ├── 采集 n-step rollout
    │   ├── 每 10 步执行 A2C 更新（GAE + 策略梯度 + 价值回归 + 熵正则）
    │   └── 返回 episode 统计（reward, completed, oom, sla, losses...）
    │
    ├── 写入训练日志
    │
    ├── 每 5 episode：打印训练进度
    │
    ├── 每 50 episode：验证评估
    │   ├── evaluate(val_env, agent, 5回合, 确定性策略)
    │   ├── 写入验证日志
    │   └── 若 val_reward 创新高 → 保存 best 模型
    │
    └── 每 100 episode：保存检查点

├── 保存 final 模型
├── 加载 best 模型 → 测试集评估
├── 写入 test_{ts}.jsonl
└── 打印测试结果
```

### 7.3 A2C 日志系统

| 特性 | 说明 |
|------|------|
| **时间戳标识** | 每次训练使用 `YYYYMMDD_HHMMSS` 时间戳，避免多次运行日志冲突 |
| **JSONL 格式** | 每行一条 JSON 记录，便于流式读取和增量分析 |
| **配置快照** | 训练开始时保存完整超参数到 `config_{ts}.yaml`，确保可复现 |
| **三阶段日志** | `train_{ts}.jsonl` / `val_{ts}.jsonl` / `test_{ts}.jsonl` 分离记录 |
| **完整指标** | 训练日志包含 reward, completed, oom, sla, cost, policy_loss, value_loss, entropy |

**训练日志字段**：

```json
{
  "timestamp": "20260422_221219",
  "episode": 100,
  "reward": -114.88,
  "steps": 200,
  "completed": 2010,
  "oom": 0,
  "sla_violations": 1333,
  "total_cost": 5.46,
  "policy_loss": -0.055,
  "value_loss": 0.832,
  "entropy": 1.404
}
```

### 7.4 可视化系统

#### DQN 可视化（6 图 + 仪表盘）

| 图表 | 文件 | 内容 |
|------|------|------|
| 奖励曲线 | `reward_curve.png` | Episode Reward + MA(20) |
| 损失曲线 | `loss_curve.png` | DQN Loss |
| Epsilon 衰减 | `epsilon_decay.png` | 探索率衰减 |
| 吞吐量 | `throughput_curve.png` | 每回合完成请求数 |
| 违规曲线 | `violations_curve.png` | OOM + SLA 违规 |
| 汇总仪表盘 | `training_dashboard.png` | 6 合 1 大图 |

#### A2C 可视化（9 图 + 仪表盘）

| 图表 | 文件 | 内容 |
|------|------|------|
| 奖励曲线 | `reward_curve.png` | Episode Reward + MA(20) |
| 损失曲线 | `loss_curves.png` | Policy Loss + Value Loss 双子图 |
| 策略熵 | `entropy_curve.png` | 探索强度变化 |
| 吞吐量 | `throughput_curve.png` | 完成请求数 |
| 成本曲线 | `cost_curve.png` | 运营成本 |
| 违规曲线 | `violations_curve.png` | OOM + SLA 违规 + MA(20) |
| 验证奖励 | `val_reward_curve.png` | 验证集 Reward ± Std |
| 验证指标 | `val_metrics.png` | 验证集 4 指标 2×2 子图 |
| 汇总仪表盘 | `training_dashboard.png` | 2×3 大图 |

A2C 可视化支持 `--timestamp` 参数选择特定训练运行，不指定则自动选取最新日志。

---

## 8. 训练结果与分析

### 8.1 A2C 训练过程

#### 训练曲线关键阶段

| 阶段 | Episode | Reward | OOM | Entropy | 分析 |
|------|---------|--------|-----|---------|------|
| 初始探索 | 0-5 | -5592 ~ -3809 | 1274-1811 | 3.55 | 高探索，大量 OOM |
| 快速学习 | 5-20 | -3809 → -116 | 1274 → 0 | 3.24 → 0.99 | 迅速学会避免 OOM |
| 稳定收敛 | 20-100 | -116 ± 5 | 0-30 | 1.0-1.5 | OOM 基本消除，SLA 为主瓶颈 |
| 持续优化 | 100-500 | -114 ± 5 | 0-20 | 1.5-2.1 | 高熵探索，策略稳健 |

#### 测试集最终结果

| 指标 | 值 |
|------|-----|
| reward_mean | -115.82 |
| reward_std | 3.07 |
| completed_mean | 2010.0 |
| oom_mean | 0.0 |
| sla_violations_mean | 1340.7 |
| total_cost_mean | 6.1341 |

### 8.2 训练稳定性问题与解决

**问题**：初次训练（无稳定性措施）在 Episode 37-50 发生崩溃，reward 暴跌至 -12,575,066 并锁死。

**根因分析**：

```
OOM 惩罚 (-10/次) × 高 OOM (2500+/步)
    → 单步 reward 达 -25,000+
        → Value MSE Loss 在十亿级 return 上计算
            → 梯度爆炸，价值网络崩坏
                → 策略坍塌（entropy 3.5 → 0.003）
                    → 确定性选择同一动作
                        → 100% OOM 死循环
```

**解决措施与效果**：

| 措施 | 修改位置 | 效果 |
|------|----------|------|
| 奖励裁剪 [-50, 50] | `train_a2c.py:59-60` | OOM 惩罚从 -25000 截断到 -50，保护训练信号 |
| 优势值裁剪 [-10, 10] | `a2c.py:134-135` | 标准化后再裁剪，防止极端优势值 |
| Huber Loss | `a2c.py:147` | 替代 MSE，对异常回报值更鲁棒 |
| 熵系数 0.01→0.05 | `default.yaml` | 策略熵维持在 1.5-2.1，防止坍塌 |
| 学习率 7e-4→3e-4 | `default.yaml` | 减小更新步长，提升稳定性 |
| Episode 缩短 500→200 | `default.yaml` | 加快迭代，减少单 episode 累积异常 |
| 训练量 2000→500 ep | `default.yaml` | 总训练步数减少 8 倍 |

### 8.3 当前性能瓶颈

Reward 在 -115 附近收敛但未进一步提升，主要瓶颈分析：

| 奖励分量 | 估算贡献 | 占比 | 瓶颈说明 |
|----------|---------|------|----------|
| `R_sla` | -134 × 0.1 = **-13.4** | 57% | SLA 违规（~1340次/episode）贡献最大负奖励 |
| `R_throughput` | +201 × 0.3 = **+60.3** | - | 已达较高水平，提升空间有限 |
| `R_latency` | ~-1.5 | - | 相对较小 |
| `R_utilization` | ~-3.0 | - | 利用率偏离 70% 的惩罚 |
| `R_cost` | ~-0.6 | - | 运营成本较低 |

**SLA 违规难以降低的原因**：
- SLA 阈值固定为 200ms，环境模拟的延迟基准为 `uniform(50, 500)`，约 60% 请求天然超过 200ms
- Agent 控制的参数（池分配、并发度、卸载策略）对单请求延迟影响有限
- 动作空间粒度较粗（36 个离散动作），无法精细控制每个请求的延迟

**潜在优化方向**：
1. 调整奖励权重：降低 SLA 权重或改用对数惩罚 `-log(1 + sla_violations)`
2. 增加动作空间粒度：提供更多池配置选项
3. 环境改进：引入请求优先级感知的调度策略

---

## 9. 算法对比

### 9.1 架构对比

| 维度 | Dueling Double DQN | A2C |
|------|-------------------|-----|
| **方法类别** | 值函数方法 | 策略梯度方法 |
| **学习目标** | Q(s,a) | π(a\|s) + V(s) |
| **网络输出** | 36 维 Q 值 | 36 维 logits + 1 维 V(s) |
| **动作选择** | ε-greedy（Q 最大 + 随机） | 策略采样（探索内生于策略） |
| **参数量** | ~110K | ~120K |
| **隐藏层** | [256,256] + LN + Dropout | [256,256] + LN |
| **探索机制** | ε 指数衰减 | 熵正则化（系数 0.05） |
| **数据效率** | 高（经验回放复用） | 低（on-policy 单次使用） |
| **训练范式** | off-policy | on-policy |
| **更新频率** | 每步更新 | 每 n_steps=10 步更新 |
| **价值损失** | Huber Loss | Huber Loss |
| **梯度裁剪** | max_norm=10.0 | max_norm=0.5 |

### 9.2 训练流程对比

| 维度 | DQN | A2C |
|------|-----|-----|
| 总 Episode 数 | 2000 | 500 |
| 每 Episode 步数 | 500 | 200 |
| 总交互步数 | 1,000,000 | 100,000 |
| 经验存储 | ReplayBuffer (100K) | RolloutBuffer (n-step) |
| 验证机制 | 无 | 每 50 ep 验证，保存 best 模型 |
| 测试机制 | 无 | 训练结束后加载 best 做测试 |
| 日志系统 | 单文件 JSONL | 时间戳分离（train/val/test）+ 配置快照 |
| GPU 支持 | 有 | 有（auto 检测 + CUDA 信息打印） |

### 9.3 理论优劣

| 维度 | DQN 优势 | A2C 优势 |
|------|---------|---------|
| 样本效率 | 经验回放复用数据，单样本贡献高 | — |
| 训练稳定性 | 目标网络提供稳定学习目标 | — |
| 离散动作空间 | Q 值穷举最优动作，天然适合 | 需从分布采样，离散空间不如 Q 学习直接 |
| 连续动作扩展 | 需 DDPG/SAC 等变体 | 自然扩展到连续空间 |
| 策略表达 | — | 直接建模 π(a\|s)，可学习随机策略 |
| 探索质量 | — | 熵正则化提供持续、自适应探索 |
| 多目标平衡 | — | 策略梯度更灵活地平衡多目标 |

---

## 10. 关键设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| 环境建模 | 3 池模型（短/长/混合） | 模拟真实 LLM 服务的多优先级调度 |
| 动作空间 | 36 离散动作（3×3×4 分解） | 平衡细粒度控制与搜索效率 |
| 状态编码 | 14 维紧凑向量 | 每池 3 维 + 全局 5 维，信息密度高 |
| 时间编码 | 正弦/余弦 | 保持周期连续性，00:00 和 23:59 编码接近 |
| 请求路由 | 基于 gamma 的概率路由 | 灵活的流量分配，由 Agent 学习最优分布 |
| DQN 架构 | Dueling + Double | 分离状态价值与动作优势，避免过估计 |
| A2C 架构 | 共享骨干 + 独立头 | 减少参数量，共享特征提取 |
| 奖励裁剪 | [-50, 50] | 防止 OOM 惩罚主导训练信号 |
| 优势估计 | GAE (λ=0.95) | 在偏差和方差间取得平衡 |
| 探索策略 | 熵正则化 (coef=0.05) | 自适应探索，替代手动 ε 衰减 |
| 日志系统 | 时间戳 JSONL + 配置快照 | 多次实验不冲突，可精确复现 |
| 验证机制 | 定期验证 + best 模型保存 | 防止过拟合，保留最优策略 |

---

## 11. 项目路线图

### 11.1 已完成

- [x] Gymnasium 兼容 RL 环境（14 维状态 / 36 离散动作 / 多目标奖励）
- [x] 仿真数据集生成（集群拓扑 + 工作负载流）
- [x] Dueling Double DQN 实现（含经验回放 + 软/硬目标网络更新）
- [x] A2C 实现（含 GAE + 熵正则 + 训练稳定性措施）
- [x] 训练脚本 + 日志系统 + 可视化
- [x] GPU/CUDA 支持
- [x] 训练/验证/测试分离评估

### 11.2 待实现

- [ ] PPO 算法
- [ ] QMIX 多 Agent 算法
- [ ] 多目标优化 / 迁移学习
- [ ] 单元测试 / 集成测试
- [ ] 大规模集群测试（数百节点）

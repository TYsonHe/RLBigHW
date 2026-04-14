# RLBigHW: 基于强化学习的大型语言模型资源优化

## 项目概述

本项目致力于使用强化学习（Reinforcement Learning, RL）技术优化大型语言模型（LLM）在异构基础设施上的资源分配。系统旨在智能地将 LLM 实例分配到具有不同配置（CPU、GPU、内存）和地理位置的机器上，以实现性能和效率的最大化。

### 核心问题

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LLM 推理请求洪流                                │
│                        (动态变化、不可预测)                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RL 智能调度器 (Agent)                            │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐    │
│   │  状态观测器  │    │   策略网络   │    │    奖励计算器          │    │
│   │  (State)    │───▶│  (Policy)   │───▶│    (Reward Function)   │    │
│   └─────────────┘    └─────────────┘    └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │  边缘节点    │  │   云端节点    │  │  混合节点    │
         │  (低延迟)    │  │  (高算力)    │  │  (成本优化)   │
         │  RTX 4090    │  │  A100-80GB   │  │  L40S-48GB   │
         │  北京        │  │   上海        │  │   广州        │
         └──────────────┘  └──────────────┘  └──────────────┘
```

**目标**：在满足 SLA 约束的前提下，最小化运营成本，最大化资源利用率。

## 问题背景

大规模部署 LLM 面临以下挑战：

- **硬件异构性**：不同配置的计算节点（CPU、GPU、内存）
- **资源地理分布**：跨地域的资源部署导致网络延迟差异
- **动态工作负载**：请求模式随时间变化的波动性
- **延迟要求差异**：不同应用对响应时间的敏感度不同
- **成本效率约束**：需要在性能和运营成本间取得平衡

传统的静态分配策略难以应对这些复杂、动态的环境需求。

## 形式化定义

### 状态空间 (State Space)

状态空间定义为 $S_t = (N_t, W_t, G_t)$，其中：

**节点状态** $N_t \in \mathbb{R}^{n \times d_n}$：

$$
N_t[i] = [\text{cpu}_i, \text{mem}_i, \text{gpu}_i, \text{gpu\_mem}_i, \text{queue}_i, \text{latency}_{i \to j}, \text{cost}_i]
$$

- $n$: 节点数量（可变）
- $\text{cpu}_i, \text{mem}_i, \text{gpu}_i, \text{gpu\_mem}_i \in [0, 1]$: 归一化资源使用率
- $\text{queue}_i \in \mathbb{N}$: 当前队列长度
- $\text{latency}_{i \to j} \in \mathbb{R}^+$: 节点间网络延迟矩阵
- $\text{cost}_i \in \mathbb{R}^+$: 单位时间运营成本

**工作负载状态** $W_t \in \mathbb{R}^{k \times d_w}$：

$$
W_t[j] = [\text{model}_j, \text{input\_len}_j, \text{output\_len}_j, \text{priority}_j, \text{sla}_j, \text{arrival\_time}_j]
$$

- $k$: 等待调度的请求数量
- $\text{model}_j \in \{1, 2, 3\}$: 模型类型 one-hot 编码
- $\text{input\_len}_j, \text{output\_len}_j \in \mathbb{N}$: token 数量
- $\text{priority}_j \in \{1, 2, 3\}$: 优先级（高/中/低）
- $\text{sla}_j \in \mathbb{R}^+$: SLA 延迟约束（毫秒）

**全局状态** $G_t \in \mathbb{R}^{d_g}$：

$$
G_t = [\text{hour\_of\_day}, \text{day\_of\_week}, \text{avg\_queue}, \text{total\_pending}]
$$

### 动作空间 (Action Space)

基于您的需求，为了同时涵盖“按配置分组”、“实例数量范围决策”、“节点配置比例分配”以及“请求比例分配”，并且**兼容所有 RL 算法（DQN、A2C、PG、QMIX）**，我为您设计了一套**“统一分组参数化动作空间”**。
这种设计的核心思想是：**引入“虚拟配置组”的概念，将原本针对单个物理节点的细粒度决策，升维为针对“特性分组”的宏观决策**。

### 一、 动作空间整体架构示意

假设我们将集群划分为 $G$ 个配置组（例如 $G=3$，分别对应：高算力组、均衡型组、边缘低延迟组），动作空间被结构化为三个并行的子空间：

```text
                      [ RL 智能调度器 Policy 输出 ]
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│          统一分组参数化动作空间 (Grouped Parameterized Action Space)     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. 节点配置分组比例 (α) —— 决定“资源按什么比例分配”              │   │
│  │    (解决：节点按配置关系分组，资源配置按节点比例分配)             │   │
│  │    ┌───────────┐   ┌───────────┐   ┌───────────┐               │   │
│  │    │   α_1     │   │   α_2     │   │   α_3     │  (约束: Σα=1) │   │
│  │    │ 40% 节点  │ + │ 40% 节点  │ + │ 20% 节点  │               │   │
│  │    │ A100-80GB │   │ L40S-48GB │   │ RTX 4090  │               │   │
│  │    └───────────┘   └───────────┘   └───────────┘               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                     │
│                                   ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 2. 实例启动数量决策 (β) —— 决定“每组启动多少实例”                │   │
│  │    (解决：实例数量是一个范围，而非简单的开/关)                    │   │
│  │    ┌───────────┐   ┌───────────┐   ┌───────────┐               │   │
│  │    │   β_1     │   │   β_2     │   │   β_3     │  (约束: 范围) │   │
│  │    │ 启动 5 个 │   │ 启动 3 个 │   │ 启动 2 个 │  [Min, Max]  │   │
│  │    │  实例     │   │  实例     │   │  实例     │               │   │
│  │    └───────────┘   └───────────┘   └───────────┘               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                     │
│                                   ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 3. 请求流量路由比例 (γ) —— 决定“请求按什么比例分配”              │   │
│  │    (解决：分配到节点的请求比例)                                   │   │
│  │    ┌───────────┐   ┌───────────┐   ┌───────────┐               │   │
│  │    │   γ_1     │   │   γ_2     │   │   γ_3     │  (约束: Σγ=1) │   │
│  │    │ 20% 流量  │ + │ 50% 流量  │ + │ 30% 流量  │               │   │
│  │    │ (高优请求)│   │ (常规请求)│   │ (容灾请求)│               │   │
│  │    └───────────┘   └───────────┘   └───────────┘               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
           【环境底层执行】将逻辑动作映射为具体的物理节点分配
```

---

### 二、 形式化数学定义

设定集群被划分为 $G$ 个配置组（每组绑定唯一的硬件配置 $\text{Config}_g = \{\text{CPU}_g, \text{MEM}_g, \text{GPU}_g\}$），总动作空间可以表示为拼接向量：
$$ A_t = [\boldsymbol{\alpha}_t, \boldsymbol{\beta}_t, \boldsymbol{\gamma}_t] \in \mathbb{R}^{3G} $$
各分量的严格定义如下：

#### 1. 节点配置分组比例 $\boldsymbol{\alpha}_t \in \Delta^{G-1}$

- **定义**：$\boldsymbol{\alpha}_t = [\alpha_1, \alpha_2, ..., \alpha_G]$，其中 $\Delta^{G-1}$ 表示 $G$ 维单纯形（Simplex）。
- **约束**：$\sum_{g=1}^{G} \alpha_g = 1$ 且 $\alpha_g \ge 0$。
- **物理意义**：决定当前时刻，整个集群的 $N_{\text{total}}$ 个物理节点中，有 $\alpha_g \times N_{\text{total}}$ 个节点被划分为第 $g$ 组，并采用配置 $\text{Config}_g$。

#### 2. 实例启动数量 $\boldsymbol{\beta}_t \in [\beta_{\min}, \beta_{\max}]^G$

- **定义**：$\boldsymbol{\beta}_t = [\beta_1, \beta_2, ..., \beta_G]$。
- **约束**：$\forall g, \beta_g \in [\beta_{\min}, \beta_{\max}]$，且 $\beta_g \in \mathbb{Z}^+$。
- **物理意义**：决定第 $g$ 组节点中，实际启动的 LLM 推理实例数量。例如设置 $\beta_{\min}=0, \beta_{\max}=10$，模型输出 4.2，经过后处理取整为 4。

#### 3. 请求路由比例 $\boldsymbol{\gamma}_t \in \Delta^{G-1}$

- **定义**：$\boldsymbol{\gamma}_t = [\gamma_1, \gamma_2, ..., \gamma_G]$。
- **约束**：$\sum_{g=1}^{G} \gamma_g = 1$ 且 $\gamma_g \ge 0$。
- **物理意义**：当一批请求洪流到达时，将 $\gamma_g$ 比例的请求路由到第 $g$ 组的实例队列中。

---

### 三、 各类 RL 算法的统一适配方案

这个设计之所以“完备”，是因为它以**连续向量的形式定义**，但可以非常自然地**离散化**，从而完美兼容四种算法：
| 算法 | 网络输出层设计 | 动作解析与后处理机制 |
| :--- | :--- | :--- |
| **A2C / PG**<br>_(天然契合)_ | **3个独立的输出头**：<br>1. $\boldsymbol{\alpha}$ 头：输出 $G$ 个 logits，过 `Softmax`<br>2. $\boldsymbol{\beta}$ 头：输出 $G$ 个值，过 `Sigmoid` 并缩放到 $[\beta_{\min}, \beta_{\max}]$<br>3. $\boldsymbol{\gamma}$ 头：输出 $G$ 个 logits，过 `Softmax` | 直接获得连续动作，$\beta$ 分量通过 `Round()` 取整后送入环境。 |
| **DQN**<br>_(离散化适配)_ | 将连续动作空间**离散化**为 Bins。假设 $G=3$，$\alpha$ 分5档，$\beta$ 分4档(0,3,6,9)，$\gamma$ 分5档。<br>总动作维度 = $5^3 \times 4^3 \times 5^3 = 8,000,000$ (维度爆炸！) | **优化方案（参数化 DQN）**：不输出笛卡尔积，而是输出 $3G$ 个连续值，但使用 **Gumbel-Softmax** 技巧使其可导，或者在 DQN 中使用 **Multi-Discrete** 动作空间（即将 $\alpha, \beta, \gamma$ 拆分为 $3G$ 个独立的离散动作臂）。 |
| **QMIX**<br>_(多智能体适配)_ | **完美契合 CTDE 架构**：每个配置组 $g$ 作为一个 Agent。<br>每个 Agent 的局部动作 $a_g = (\alpha_g, \beta_g, \gamma_g) \in \mathbb{R}^3$。 | - 每个 Agent 输出本组的节点占比、实例数和期望流量。<br>- 环境端收集所有 Agent 的 $\alpha_g$ 并做归一化 $\sum \alpha_g = 1$。<br>- Mixer 网络基于全局状态融合各 Agent 的 Q 值。 |

---

### 四、 环境侧的约束与安全映射

为了防止 RL 模型输出导致物理上不可行的动作（例如：将 90% 的节点分配给高配组，但只启动 0 个实例），在 `step(action)` 接收到 $A_t$ 后，需要加入一层**安全映射器**：

```python
def safe_action_mapping(action_vector, total_nodes, group_configs):
    alpha, beta, gamma = split_action(action_vector)

    # 1. 节点数量取整与修正 (防止浮点数误差导致总节点数差1)
    node_counts = [int(a * total_nodes) for a in alpha]
    node_counts[-1] = total_nodes - sum(node_counts[:-1]) # 尾差修正

    # 2. 实例数量边界钳位与校验 (实例数不能超过该组分配到的节点数)
    for g in range(G):
        beta[g] = int(np.clip(round(beta[g]), 0, node_counts[g]))

    # 3. 流量比例归一化 (防止 Softmax 精度问题)
    gamma = gamma / np.sum(gamma)

    # 4. 资源物理约束校验 (结合 Readme 中的惩罚机制)
    # 如果某组分配了 DeepSeek-70B 但节点数<1，直接触发 R_penalty = -10.0

    return mapped_physical_action
```

### 五、 该设计的核心优势

1. **彻底解决了动作空间异构性问题**：以前 DQN 只能选节点，PG 只能选比例。现在大家都在同一个数学空间 $[\boldsymbol{\alpha}, \boldsymbol{\beta}, \boldsymbol{\gamma}]$ 里博弈，**算法对比绝对公平**。
2. **契合真实运维逻辑**：运维人员通常不会说“把请求发给 node_54”，而是说“把 40% 的流量发给 A100 集群，那边起 5 个实例”。该动作空间直接对应了人类的运维语义。
3. **维度可控**：不论集群有 50 个节点还是 5000 个节点，只要分组数 $G$ 固定（如 $G=3$），动作空间的维度永远是 $3G = 9$ 维，**完美规避了节点规模变化导致的动作维度爆炸问题**。

### 奖励函数 (Reward Function)

**多目标加权奖励**：

$$
R_t = w_1 \cdot R_{\text{throughput}} + w_2 \cdot R_{\text{latency}} + w_3 \cdot R_{\text{util}} + w_4 \cdot R_{\text{cost}} + w_5 \cdot R_{\text{sla}}
$$

**各分量定义**：

1. **吞吐量奖励**：
   $$R_{\text{throughput}} = \frac{\text{completed\_requests}_t}{\text{time\_step}_t} \cdot \text{norm\_factor}$$

2. **延迟惩罚**：
   $$R_{\text{latency}} = -\frac{1}{k}\sum_{j=1}^{k} \left(\frac{\text{latency}_j}{\text{sla}_j}\right)^2$$

3. **资源利用率奖励**：
   $$R_{\text{util}} = \frac{1}{n}\sum_{i=1}^{n} \left(1 - |\text{target\_util}_i - \text{actual\_util}_i|\right)$$

4. **成本惩罚**：
   $$R_{\text{cost}} = -\frac{\sum_i \text{cost}_i \cdot \text{active\_time}_i}{\text{total\_revenue}_t}$$

5. **SLA 达成奖励**：
   $$R_{\text{sla}} = \frac{\text{requests\_within\_sla}_t}{\text{total\_requests}_t}$$

**约束惩罚**：

$$
R_{\text{penalty}} =
\begin{cases}
-10.0, & \text{if OOM (显存溢出)} \\
-5.0, & \text{if queue overflow (队列溢出)} \\
-2.0, & \text{if SLA violation > 5\%}
\end{cases}
$$

## 技术方案

### 强化学习框架

#### 状态空间设计

- **节点配置**：CPU/GPU 型号、内存大小、地理位置、节点网络速率
- **工作负载特征**：请求类型、输入长度、模型类型
- **系统状态**：队列长度、资源利用率、网络状况
- **时间因素**：周期性流量模式、请求时间分布

#### 动作空间设计

- **实例部署**：选择部署节点
- **资源分配**：CPU/GPU 核心分配、内存分配
- **弹性伸缩**：实例数量动态调整
- **请求路由**：智能请求分配

#### 奖励函数设计

多目标优化函数，平衡以下因素：

```
奖励 = w1×吞吐量 + w2×(1/延迟) + w3×资源利用率 + w4×(1/成本) + w5×(1/错误率)
```

其中权重系数 (w1-w5) 可根据业务需求动态调整（支持可学习权重策略）。

## 数据生成引擎算法详解

### 工作负载到达过程：泊松过程

请求到达遵循**非齐次泊松过程** (Non-homogeneous Poisson Process)：

$$
\lambda(t) = \lambda_{\text{base}} \cdot (1 + A \cdot \sin(2\pi \frac{t - \phi}{T}))
$$

- $\lambda_{\text{base}}$: 基础到达率（请求/秒）
- $A$: 潮汐效应振幅（0.5 表示±50% 波动）
- $\phi$: 相位偏移（早 9 点高峰）
- $T$: 周期（24 小时）

**实现伪代码**：

```python
def generate_arrival_times(duration_hours: float, lambda_base: float) -> List[float]:
    """生成符合潮汐效应的请求到达时间序列。"""
    arrivals = []
    t = 0.0
    while t < duration_hours * 3600:  # 转换为秒
        # 计算当前时刻的瞬时到达率
        hour_of_day = (t / 3600) % 24
        lambda_t = lambda_base * (1 + 0.5 * np.sin(2 * np.pi * (hour_of_day - 9) / 24))

        # 生成下一个到达的时间间隔（指数分布）
        delta_t = np.random.exponential(1 / lambda_t)
        t += delta_t
        arrivals.append(t)

    return arrivals
```

### Token 长度分布：对数正态分布

输入/输出 token 长度遵循**对数正态分布** (Log-Normal Distribution)：

$$
\ln(X) \sim \mathcal{N}(\mu, \sigma^2)
$$

参数设定：

- **输入长度**: $\mu=7.5, \sigma=1.2$（对应中位数~1800 tokens，长尾至 4096）
- **输出长度**: $\mu=6.0, \sigma=0.8$（对应中位数~400 tokens）

```python
def generate_token_lengths(num_samples: int, dist_type: str) -> np.ndarray:
    """生成符合长尾分布的 token 长度。"""
    params = {
        'input': {'mu': 7.5, 'sigma': 1.2, 'max': 4096},
        'output': {'mu': 6.0, 'sigma': 0.8, 'max': 2048}
    }
    p = params[dist_type]
    samples = np.random.lognormal(mean=p['mu'], sigma=p['sigma'], size=num_samples)
    return np.clip(samples, 50, p['max']).astype(int)
```

### 资源消耗模型

**显存需求计算**：

$$
\text{VRAM}_{\text{req}} = \text{VRAM}_{\text{model}} + \text{VRAM}_{\text{kv-cache}} \times (\text{input\_len} + \text{output\_len})
$$

| 模型         | 基础显存 | KV Cache per token |
| ------------ | -------- | ------------------ |
| LLaMA3-7B    | 14 GB    | 2 KB               |
| Qwen-14B     | 28 GB    | 4 KB               |
| DeepSeek-70B | 140 GB   | 16 KB              |

**推理延迟估算**：

$$
\text{latency} = \text{latency}_{\text{base}} + \text{latency}_{\text{per-token}} \times \text{output\_len}
$$

| GPU 型号                | A100-80GB | L40S-48GB | RTX 4090-24GB |
| ----------------------- | --------- | --------- | ------------- |
| LLaMA3-7B (ms/token)    | 2.5       | 3.2       | 4.0           |
| Qwen-14B (ms/token)     | 4.0       | 5.5       | 8.0           |
| DeepSeek-70B (ms/token) | 12.0      | 18.0      | N/A           |

## RL 算法网络架构详解

### DQN (Deep Q-Network)

**网络架构**：

```
输入 (state_vector)
       │
       ▼
┌───────────────────┐
│  Fully Connected  │  (512 单元)
│  + LayerNorm      │
│  + ReLU           │
└───────────────────┘
       │
       ▼
┌───────────────────┐
│     Dropout       │  (p=0.1)
└───────────────────┘
       │
       ▼
┌───────────────────┐
│  Fully Connected  │  (256 单元)
│  + ReLU           │
└───────────────────┘
       │
       ▼
┌───────────────────┐
│  Fully Connected  │  (num_nodes 单元)
│   (Q-value 输出)   │
└───────────────────┘
```

**超参数配置**：
| 超参数 | 值 | 说明 |
|--------|-----|------|
| 学习率 | 1e-4 | Adam 优化器 |
| 折扣因子 γ | 0.99 | 未来奖励折扣 |
| 缓冲池大小 | 100,000 | 经验回放容量 |
| 批次大小 | 256 | 每次梯度更新样本数 |
| ε_start | 1.0 | 初始探索率 |
| ε_end | 0.01 | 最小探索率 |
| ε_decay | 50,000 | 探索率衰减步数 |
| target_update | 1,000 | 目标网络更新频率 |

**Loss 函数**：

$$
L(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{\text{target}}(s', a') - Q(s, a)\right)^2\right]
$$

### A2C (Advantage Actor-Critic)

**网络架构**（共享骨干）：

```
输入 (state_vector)
       │
       ▼
┌───────────────────┐
│  Shared Backbone  │  (512 单元，ReLU)
└───────────────────┘
       │
       ├─────────────┬─────────────┐
       ▼             ▼             ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│   Actor   │ │  Critic   │ │  Value     │
│ (μ, σ)    │ │   (V)     │ │  (基线)    │
└───────────┘ └───────────┘ └───────────┘
```

**优势估计 (GAE)**：

$$
A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots
$$

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**超参数配置**：
| 超参数 | 值 | 说明 |
|--------|-----|------|
| 学习率 (actor) | 3e-4 | Adam 优化器 |
| 学习率 (critic) | 1e-3 | Adam 优化器 |
| γ | 0.99 | 折扣因子 |
| λ | 0.95 | GAE 平滑参数 |
| 熵系数 | 0.01 | 鼓励探索 |
| value_loss_coef | 0.5 | 价值损失权重 |
| max_grad_norm | 0.5 | 梯度裁剪 |

### QMIX (多 Agent 协作)

**架构组件**：

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent 网络 (每个节点)                      │
│  local_obs → GRU(64) → FC(128) → Q_agent(local_action)      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Mixing Network                          │
│  Q_total = f_mix(Q1, Q2, ..., Qn; hypernetwork_weights)      │
│  约束：∂Q_total/∂Q_a ≥ 0 (单调性约束)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Hyper Network                             │
│  global_state → FC(256) → ReLU → FC(n_agents × weights)     │
└─────────────────────────────────────────────────────────────┘
```

### 算法对比总结

| 算法         | 类型     | 适用场景     | 优势                           |
| ------------ | -------- | ------------ | ------------------------------ |
| **策略梯度** | 策略优化 | 连续动作空间 | 直接优化策略，适合高维动作空间 |
| **A2C**      | 策略优化 | 并行环境     | 样本效率高，减少训练方差       |
| **DQN**      | 值优化   | 离散动作空间 | 稳定可靠，经验回放机制         |
| **QMIX**     | 多 Agent | 分布式决策   | 集中训练分散执行，适合集群环境 |

### 仿真环境

#### 数据集设计

数据集包含以下关键表结构：

**节点配置信息**
| 字段 | 描述 |
|------|------|
| 节点 ID | 唯一标识符 |
| CPU 使用率 (%) | 当前 CPU 利用率 |
| 内存使用 (MB) | 当前内存使用量 |
| GPU 使用率 (%) | 当前 GPU 利用率 |
| GPU 内存使用 (MB) | GPU 显存使用量 |
| 磁盘 IO(ops/s) | 磁盘操作频率 |
| 网络带宽 (Mbps) | 可用网络带宽 |
| 节点位置 | 地理位置 |
| 实例数量 | 部署的实例数 |
| 实例类型 | 硬件配置类型 |

**请求特征**
| 字段 | 描述 |
|------|------|
| 请求 ID | 唯一标识符 |
| 请求类型 | 文本生成/摘要生成/问答等 |
| 输入长度 | 输入 token 数量 |
| 输出长度 | 输出 token 数量 |
| 模型类型 | LLaMA-7B/GPT-3 等 |
| 请求时间戳 | 请求到达时间 |
| 处理时间 (ms) | 实际处理耗时 |
| 响应延迟 (ms) | 端到端延迟 |
| 能耗 (kWh) | 请求处理能耗 |
| 成本 (元) | 请求处理成本 |
| 错误率 (%) | 处理失败率 |

完整数据集模板见：[Datasets/simulation_dataset_template.csv](Datasets/simulation_dataset_template.csv)

要求：

- 模型类型固定为 1-3 个型号
- 节点配置中，CPU 固定描述为多少 cores，内存固定描述为多少 GB，磁盘固定描述为多少 GB，GPU 型号固定为 1-3 个目前常用型号

## 核心功能

1. **异构资源管理**：优化不同硬件配置的资源分配
2. **地理感知调度**：根据节点位置智能分配请求，降低延迟
3. **动态弹性伸缩**：根据负载自动调整资源分配
4. **成本效率优化**：平衡性能与运营成本
5. **多目标优化**：同时优化多个可能冲突的目标
6. **容错机制**：自动处理节点故障和网络波动

## 数据集

数据集存储在 `Datasets/` 目录：

- `invoke_data2.csv`：工作负载模式和调用数据
- `load_test_results.csv`：负载测试性能指标，包含：
  - 资源利用率（CPU、内存、GPU）
  - 实例数量
  - 吞吐量（RPS）
  - 延迟指标（P50、P95、P99）
  - 时间戳
- `simulation_dataset_template.csv`：仿真数据集模板

## 评估指标

### 核心性能指标 (KPIs)

| 指标类别   | 具体指标     | 计算方式                                               | 目标值   |
| ---------- | ------------ | ------------------------------------------------------ | -------- |
| **延迟**   | 平均响应时间 | $\frac{1}{N}\sum_{i=1}^{N} \text{latency}_i$           | < 200ms  |
|            | P95 延迟     | 95% 请求的延迟上限                                     | < 500ms  |
|            | P99 延迟     | 99% 请求的延迟上限                                     | < 1000ms |
| **吞吐**   | 峰值 RPS     | max(requests per second)                               | > 10,000 |
|            | 稳定态 RPS   | avg(requests per second)                               | > 5,000  |
| **效率**   | GPU 利用率   | $\frac{\text{active\_time}}{\text{total\_time}}$       | 60-80%   |
|            | 能效比       | RPS / Watt                                             | > 100    |
| **成本**   | 单次推理成本 | 总成本 / 总请求数                                      | < ¥0.001 |
| **稳定性** | SLA 达成率   | $\frac{\text{within\_sla}}{\text{total}} \times 100\%$ | > 99%    |
|            | 故障恢复时间 | 从故障到恢复的时间                                     | < 30s    |

## 评估协议

### 公平对比协议

为确保不同 RL 算法的公平对比，遵循以下协议：

**1. 固定随机种子**

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

**2. 相同训练数据**

- 所有算法使用相同的 `workload_streams_train.csv`
- 相同的 `cluster_profiles_train.csv`

**3. 统一评估流程**

```
┌─────────────────────────────────────────────────────────────┐
│                    评估流程                                   │
├─────────────────────────────────────────────────────────────┤
│  1. 加载训练好的模型 checkpoint                              │
│  2. 关闭探索模式 (ε=0, deterministic policy)                │
│  3. 在 test 数据集上运行完整评估                             │
│  4. 记录所有指标：延迟、吞吐、成本、SLA                      │
│  5. 重复 3 次取平均值 (不同种子)                              │
└─────────────────────────────────────────────────────────────┘
```

**4. 基线对比**

- **随机策略**：随机选择可用节点
- **轮询策略**：Round-Robin 分配
- **最短队列**：分配到队列最短的节点
- **最小成本**：分配到成本最低的可用节点

### 验收标准

| Phase       | 验收项      | 通过标准                         |
| ----------- | ----------- | -------------------------------- |
| **Phase 1** | 数据生成器  | 生成 6 个 CSV 文件，格式符合规范 |
|             | RL 环境     | 通过 `pytest tests/test_env.py`  |
|             | 奖励函数    | 返回值范围合理，无 NaN/Inf       |
| **Phase 2** | DQN 训练    | Reward 曲线在 50k 步内上升       |
|             | 模型保存    | 可加载 checkpoint 并推理         |
|             | 评估框架    | 输出完整指标报告                 |
| **Phase 3** | A2C/PG 实现 | 收敛速度快于 DQN                 |
|             | 算法对比    | 生成对比表格和雷达图             |
| **Phase 4** | QMIX 协作   | 多节点场景下优于单 Agent         |
|             | 通信开销    | < 5% 额外延迟                    |
| **Phase 5** | 压力测试    | 10 倍流量下不崩溃                |
|             | 可视化      | 生成论文级图表                   |

## 实现路线图

1. **基础框架搭建**（当前阶段）
   - 仿真环境开发
   - 数据集生成工具
   - 基本 RL 算法实现
2. **算法优化阶段**
   - 多 Agent 系统开发
   - 自适应奖励函数
   - 迁移学习支持

## 使用说明

### RL 算法对比

| 算法         | 类型     | 适用场景     | 优势                           |
| ------------ | -------- | ------------ | ------------------------------ |
| **策略梯度** | 策略优化 | 连续动作空间 | 直接优化策略，适合高维动作空间 |
| **A2C**      | 策略优化 | 并行环境     | 样本效率高，减少训练方差       |
| **DQN**      | 值优化   | 离散动作空间 | 稳定可靠，经验回放机制         |
| **QMIX**     | 多 Agent | 分布式决策   | 集中训练分散执行，适合集群环境 |

---

## 真实世界数据集设计规范

为了确保 RL 算法在接近生产环境的条件下进行训练与评估，本项目**通过程序化生成引擎构建大规模、符合真实物理约束的仿真数据集**，并严格划分为训练集、验证集和测试集。

### 1. 硬件与模型约束（强制规范）

在生成数据时，必须严格遵循以下离散枚举值：

- **支持的模型类型（固定 3 种）**：
  1. `LLaMA3-7B` (轻量级，低延迟要求)
  2. `Qwen-14B` (中等规模，均衡型)
  3. `DeepSeek-70B` (重量级，高显存与长上下文要求)
- **支持的 GPU 型号（固定 3 种常用型号）**：
  1. `NVIDIA A100-SXM4-80GB` (高端算力)
  2. `NVIDIA L40S-48GB` (中端推理主力)
  3. `NVIDIA RTX 4090-24GB` (消费级/边缘算力)

- **资源描述规范**：
  - CPU：必须描述为核数（如 `64 cores`, `128 cores`）
  - 内存：必须描述为 GB（如 `256 GB`, `512 GB`）
  - 磁盘：必须描述为 GB（如 `2000 GB NVMe`）

### 2. 数据集构成与规模要求

> **注意**：采用中等规模模拟方案，适合快速迭代和算法验证。

生成的数据集存放在 `Datasets/generated/` 目录下，分为以下两类文件：

| 数据集类型            | 文件                         | 规模       | 说明                                        |
| --------------------- | ---------------------------- | ---------- | ------------------------------------------- |
| **集群拓扑 (训练集)** | `cluster_profiles_train.csv` | 500 个节点 | 随机组合的异构节点配置                      |
| **集群拓扑 (验证集)** | `cluster_profiles_val.csv`   | 100 个节点 | 用于训练过程中的验证                        |
| **集群拓扑 (测试集)** | `cluster_profiles_test.csv`  | 50 个节点  | 包含极端场景（单一 GPU 型号、跨地域高延迟） |
| **工作负载 (训练集)** | `workload_streams_train.csv` | ~50,000 条 | 模拟 7 天，包含工作日/周末潮汐模式          |
| **工作负载 (验证集)** | `workload_streams_val.csv`   | ~10,000 条 | 模拟 1 天，用于训练过程验证                 |
| **工作负载 (测试集)** | `workload_streams_test.csv`  | ~10,000 条 | 模拟 1 天，包含突发流量场景                 |

**核心字段说明**：

- 集群拓扑：`node_id, region, cpu_cores, mem_gb, disk_gb, gpu_model, gpu_mem_gb, cost_per_hour, network_bandwidth_mbps`
- 工作负载：`timestamp, req_id, model_type, input_tokens, output_tokens, priority, sla_deadline_ms`

---

## 开发里程碑与详细需求说明书

> **说明**：严格按照以下 5 个 Phase 迭代。每个阶段明确了交付物、接口规范和验收标准。

### Phase 1: 基础设施与数据生成底座

**目标**：不写任何 RL 代码，纯 Python/PyTorch 构建高保真数据生成器与仿真环境骨架。

- **需求 1.1：集群拓扑数据生成器**
  - **文件**：`src/utils/data_generators/cluster_generator.py`
  - **类/方法**：`class ClusterTopologyGenerator`
  - **详细要求**：
    - 读取预定义的硬件字典（A100/L40S/4090 的参数）
    - 实现 `generate(num_nodes: int, split: str)` 方法
    - 能够随机组合硬件，并根据地理位置（如北京->上海增加 20ms 基础延迟）计算网络拓扑矩阵
    - 按比例输出 `train/val/test` 三个 CSV 文件，确保条数达到 500/100/50 的标准

- **需求 1.2：工作负载流数据生成器**
  - **文件**：`src/utils/data_generators/workload_generator.py`
  - **类/方法**：`class WorkloadStreamGenerator`
  - **详细要求**：
    - 使用 `numpy` 拟合泊松过程模拟请求到达
    - Token 长度必须使用对数正态分布或伽马分布模拟真实的长尾特征（大部分请求很短，少数极长）
    - 实现潮汐效应：在早 9 点、晚 8 点注入正弦波流量高峰
    - 输出 `train/val/test` 三个 CSV，确保条数达到 5 万/1 万/1 万的标准。必须包含 lazy-writing 机制，避免大量数据直接撑爆内存

- **需求 1.3：数据加载与预处理工具**
  - **文件**：`src/utils/data_loader.py`
  - **类/方法**：`class DataPipeline`
  - **详细要求**：实现 `load_cluster(split)` 和 `get_workload_iterator(split)`。由于工作负载数据量大，迭代器需支持按批次从磁盘读取

- **需求 1.4：Gymnasium 环境核心骨架**
  - **文件**：`src/envs/cluster_env.py`
  - **类/方法**：`class LLMClusterEnv(gym.Env)`
  - **详细要求**：
    - `__init__` 接收 `DataPipeline` 提供的初始状态，定义 `observation_space` 和 `action_space`
    - `step(action)`：接收动作（节点 ID），根据当前时刻的 Workload 特征，模拟资源扣减（基于选定的模型类型和 GPU 型号计算推理时间），计算奖励

- **需求 1.5：独立奖励函数模块**
  - **文件**：`src/envs/reward_fn.py`
  - **类/方法**：`def calculate_reward(prev_state, current_state, action, config) -> float`
  - **详细要求**：实现多目标加权逻辑。若请求被分配到显存不足的节点（如将大模型分配给显存不足的 GPU），直接返回极大负惩罚

- **验收标准**：成功运行生成脚本，在 `Datasets/generated/` 下生成符合规范的 6 个 CSV 文件。随机抽样检查：GPU 型号必须属于指定的 3 种，Token 数呈现长尾分布

### Phase 2: 单 Agent 基线算法验证

**目标**：实现 DQN，跑通基于仿真数据的训练闭环。

- **需求 2.1：算法基础抽象类** (`src/algorithms/base_agent.py`)：定义 `act`, `learn`, `save`, `load` 抽象接口
- **需求 2.2：高效经验回放池** (`src/algorithms/replay_buffer.py`)：支持标准均匀采样
- **需求 2.3：DQN 算法实现** (`src/algorithms/dqn.py`)：包含 Double DQN 和 Dueling DQN 架构。输入为环境 State 张量，输出为各个节点的 Q 值
- **需求 2.4：基础训练主循环** (`scripts/train_dqn.py`)：对接 Env 和 Agent，实现 Epsilon 衰减，每 1000 步保存一次 Checkpoint
- **验收标准**：在包含 100 个节点的子集上运行 DQN，Reward 曲线在 5 万步内呈现上升趋势

### Phase 3: 进阶单智能体与调参

**目标**：引入策略梯度，对比离散与连续动作空间。

- **需求 3.1：策略梯度 (REINFORCE)** (`src/algorithms/policy_gradient.py`)
- **需求 3.2：A2C 算法** (`src/algorithms/a2c.py`)：实现 Actor-Critic 双头网络与 GAE 优势估计
- **需求 3.3：统一评估框架** (`src/utils/evaluation.py`)：关闭探索，运行完整测试集流，输出 `{平均延迟，P99 延迟，吞吐量，总成本}` 汇总字典
- **验收标准**：使用评估框架对比 DQN、PG、A2C 在相同测试集流上的表现，生成对比表格

### Phase 4: 多 Agent 分布式决策扩展

**目标**：拆解单一调度器，每个节点作为一个 Agent，使用 QMIX 协作。

- **需求 4.1：多智能体环境包装器** (`src/envs/multi_cluster_wrapper.py`)：将全局 Obs 拆分为 N 个局部 Obs，Action 也拆分为 N 个
- **需求 4.2：QMIX 组件** (`src/algorithms/qmix/components.py`)：实现 RNNAgent (处理时序请求流) 和满足单调性约束的 QMixer 网络
- **需求 4.3：QMIX 训练器** (`src/algorithms/qmix/qmix_trainer.py`)：实现 CTDE（集中训练分散执行）逻辑
- **验收标准**：在多节点环境下，验证边缘节点满载时能通过 QMIX 自发将流量路由至云端节点

### Phase 5: 评估体系与工程化收尾

**目标**：输出科研成果级别的图表与严谨的测试。

- **需求 5.1：边界与压力测试** (`tests/test_env.py`)：使用极端流量（如瞬间 10 倍峰值）测试环境是否崩溃，奖励计算是否出现 NaN
- **需求 5.2：可视化模块** (`src/utils/visualization.py`)：使用 Plotly 绘制训练 Loss 曲线、奖励移动平均线；绘制不同算法的多维雷达图（延迟/成本/吞吐/利用率）
- **验收标准**：`pytest tests/` 100% 通过，生成可直接用于论文的交互式 HTML 图表

---

## 项目结构

```
RLBigHW/
├── Datasets/
│   ├── invoke_data2.csv                 # 原始参考流量
│   ├── load_test_results.csv            # 原始参考压测
│   ├── simulation_dataset_template.csv  # 原始模板
│   └── generated/                       # 大规模仿真数据
│       ├── cluster_profiles_train.csv   # 500 条节点配置
│       ├── cluster_profiles_val.csv     # 100 条节点配置
│       ├── cluster_profiles_test.csv    # 50 条节点配置
│       ├── workload_streams_train.csv   # ~5 万条请求流
│       ├── workload_streams_val.csv     # ~1 万条请求流
│       └── workload_streams_test.csv    # ~1 万条请求流
│
├── src/
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── cluster_env.py           # 主环境类
│   │   ├── reward_fn.py             # 奖励函数
│   │   └── multi_cluster_wrapper.py # 多智能体包装器
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base_agent.py            # 抽象基类
│   │   ├── replay_buffer.py         # 回放池
│   │   ├── dqn.py                   # DQN
│   │   ├── policy_gradient.py       # PG
│   │   ├── a2c.py                   # A2C
│   │   └── qmix/                    # QMIX
│   │       ├── components.py
│   │       └── qmix_trainer.py
│   ├── utils/
│   │   ├── data_loader.py           # 数据加载管道
│   │   ├── data_generators/         # 数据生成引擎
│   │   │   ├── cluster_generator.py # 节点拓扑生成
│   │   │   └── workload_generator.py# 请求流生成
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── configs/
│       ├── dqn_config.yaml
│       └── env_config.yaml
│
├── tests/
├── scripts/                             # 执行脚本 (如 train_dqn.py)
├── Readme.md
├── CLAUDE.md
└── requirements.txt
```

## 安装说明

### 环境要求

- Python 3.10+
- CUDA 11.7+ (可选，用于 GPU 加速训练)

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd RLBigHW

# 创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 依赖说明

主要依赖包：

- `torch` - 深度学习框架
- `gymnasium` - RL 环境接口
- `numpy` - 数值计算
- `pandas` - 数据处理
- `matplotlib` / `plotly` - 可视化
- `pytest` - 测试框架

## 快速开始

### 1. 生成仿真数据集

```bash
python scripts/generate_datasets.py
```

### 2. 训练 DQN 模型

```bash
python scripts/train_dqn.py --config src/configs/dqn_config.yaml
```

### 3. 评估模型

```bash
python scripts/evaluate.py --checkpoint checkpoints/dqn_best.pt
```

## 使用示例

### 示例 1：自定义环境配置

```python
from src.envs.cluster_env import LLMClusterEnv
from src.utils.data_loader import DataPipeline

# 加载数据管道
data_pipeline = DataPipeline(data_dir="Datasets/generated")

# 创建环境实例
env = LLMClusterEnv(
    data_pipeline=data_pipeline,
    num_nodes=100,
    regions=["beijing", "shanghai", "guangzhou"],
    reward_weights={
        "throughput": 0.3,
        "latency": 0.3,
        "utilization": 0.2,
        "cost": 0.1,
        "sla": 0.1
    }
)

# 重置环境
obs = env.reset()

# 运行一个 episode
for t in range(1000):
    # 随机动作（仅用于测试）
    action = env.action_space.sample()

    # 执行动作
    next_obs, reward, done, info = env.step(action)

    if done:
        break

env.close()
```

### 示例 2：训练 DQN 模型

```python
from src.algorithms.dqn import DQNAgent
from src.envs.cluster_env import LLMClusterEnv
from src.utils.data_loader import DataPipeline
import torch

# 初始化环境和 agent
data_pipeline = DataPipeline(data_dir="Datasets/generated")
env = LLMClusterEnv(data_pipeline=data_pipeline, num_nodes=50)
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    learning_rate=1e-4,
    buffer_size=100000
)

# 训练循环
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(500):
        # ε-greedy 动作选择
        action = agent.act(state, epsilon=max(0.01, 1.0 - episode * 0.001))

        # 执行动作
        next_state, reward, done, info = env.step(action)

        # 存储到经验回放池
        agent.remember(state, action, reward, next_state, done)

        # 学习
        agent.learn(batch_size=256)

        state = next_state
        total_reward += reward

        if done:
            break

    # 每 100 集保存一次 checkpoint
    if episode % 100 == 0:
        agent.save(f"checkpoints/dqn_episode_{episode}.pt")

    print(f"Episode {episode}: total_reward={total_reward:.2f}")
```

### 示例 3：多算法对比评估

```python
from src.utils.evaluation import EvaluationFramework
from src.algorithms.dqn import DQNAgent
from src.algorithms.a2c import A2CAgent
from src.algorithms.policy_gradient import PGAgent

# 初始化评估框架
evaluator = EvaluationFramework(
    env_config={"num_nodes": 100},
    test_split="test"
)

# 加载不同算法的模型
agents = {
    "DQN": DQNAgent.load("checkpoints/dqn_best.pt"),
    "A2C": A2CAgent.load("checkpoints/a2c_best.pt"),
    "PG": PGAgent.load("checkpoints/pg_best.pt"),
}

# 运行对比评估
results = evaluator.compare_agents(agents, num_episodes=50)

# 输出对比表格
print(results.to_markdown())

# 生成可视化图表
evaluator.plot_radar_chart(results, save_path="results/algorithm_comparison.png")
evaluator.plot_training_curves(results, save_path="results/training_curves.png")
```

### 示例 4：压力测试场景

```python
from src.envs.cluster_env import LLMClusterEnv
from src.utils.data_generators.workload_generator import WorkloadStreamGenerator

# 生成突发流量场景
generator = WorkloadStreamGenerator()
burst_workload = generator.generate_burst_traffic(
    base_rate=100,  # 基础请求/秒
    burst_multiplier=10,  # 10 倍峰值
    duration_seconds=60
)

# 在突发流量下测试环境
env = LLMClusterEnv(num_nodes=50)
env.load_workload(burst_workload)

state = env.reset()
total_processed = 0
sla_violations = 0

for t in range(3600):  # 模拟 1 小时
    action = agent.act(state, epsilon=0)  # 纯利用模式
    next_state, reward, done, info = env.step(action)

    total_processed += info.get("processed_requests", 0)
    sla_violations += info.get("sla_violations", 0)

    state = next_state

print(f"总处理请求：{total_processed}")
print(f"SLA 违规次数：{sla_violations}")
print(f"SLA 达成率：{(1 - sla_violations/total_processed) * 100:.2f}%")
```

## 典型应用场景

### 场景 1：边缘计算集群调度

```
┌─────────────────────────────────────────────────────────────┐
│                    边缘计算场景                               │
├─────────────────────────────────────────────────────────────┤
│  节点类型：RTX 4090 (24GB) × 20                              │
│  地理位置：分散在 10 个城市                                     │
│  工作负载：低延迟要求的实时交互 (SLA < 100ms)                │
│  优化目标：最小化延迟，同时控制成本                          │
└─────────────────────────────────────────────────────────────┘
```

**推荐算法**：A2C（连续动作空间，快速响应）

### 场景 2：云端大规模推理集群

```
┌─────────────────────────────────────────────────────────────┐
│                    云端推理场景                               │
├─────────────────────────────────────────────────────────────┤
│  节点类型：A100-80GB × 100                                  │
│  地理位置：集中式数据中心                                    │
│  工作负载：批量推理任务，吞吐优先                            │
│  优化目标：最大化吞吐量，优化能效比                          │
└─────────────────────────────────────────────────────────────┘
```

**推荐算法**：DQN（离散动作空间，稳定收敛）

### 场景 3：混合云协同调度

```
┌─────────────────────────────────────────────────────────────┐
│                    混合云场景                                 │
├─────────────────────────────────────────────────────────────┤
│  节点类型：A100 + L40S + RTX4090 混合部署                    │
│  地理位置：边缘 + 云端协同                                    │
│  工作负载：多样化，包含实时和批量任务                        │
│  优化目标：多目标平衡，成本效率最优化                        │
└─────────────────────────────────────────────────────────────┘
```

**推荐算法**：QMIX（多 Agent 协作，集中训练分散执行）

## 配置示例

### 环境配置 (`src/configs/env_config.yaml`)

```yaml
env:
  num_nodes: 100
  regions: ["beijing", "shanghai", "guangzhou"]
  gpu_types: ["A100-80GB", "L40S-48GB", "RTX4090-24GB"]
  model_types: ["LLaMA3-7B", "Qwen-14B", "DeepSeek-70B"]

reward:
  weights:
    throughput: 0.3
    latency: 0.3
    utilization: 0.2
    cost: 0.1
    error_rate: 0.1
```

### DQN 配置 (`src/configs/dqn_config.yaml`)

```yaml
agent:
  learning_rate: 0.0001
  buffer_size: 100000
  batch_size: 256
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 10000
  target_update: 1000

network:
  hidden_dims: [256, 256, 128]
  activation: "relu"
  dropout: 0.1
```

## 评估指标详解

我们采用多维雷达图评估模型效能：

1. **经济效益**：单次推理成本 = (节点能耗成本 + 硬件折旧成本) / 总成功请求数

2. **时效性**：
   - P99 Tail 延迟（排除长尾效应干扰）
   - 平均响应时间

3. **资源效率**：加权资源利用率 = (0.4×CPU_util + 0.4×GPU_util + 0.2×MEM_util) / 总分配资源

4. **稳定性**：
   - SLA 达成率（延迟小于 200ms 的请求占比）
   - 节点过载发生率

## 贡献指南

欢迎加入我们打造下一代 AI 基础设施调度系统！请阅读 [`CLAUDE.md`](CLAUDE.md) 了解代码规范。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/Algorithm-PPO`)
3. 确保代码通过 `pytest tests/` 且覆盖率达到 80% 以上
4. 提交 PR 并附上算法在仿真环境中的性能对比截图

**注意**：所有主要更改应添加测试用例，并确保通过现有测试。

**注意**：随着项目进展，请及时更新此文件以反映实际代码结构和开发实践。

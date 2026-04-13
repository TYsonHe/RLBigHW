# RLBigHW: 基于强化学习的大型语言模型资源优化

## 项目概述

本项目致力于使用强化学习（Reinforcement Learning, RL）技术优化大型语言模型（LLM）在异构基础设施上的资源分配。系统旨在智能地将LLM实例分配到具有不同配置（CPU、GPU、内存）和地理位置的机器上，以实现性能和效率的最大化。

## 问题背景

大规模部署LLM面临以下挑战：

- **硬件异构性**：不同配置的计算节点（CPU、GPU、内存）
- **资源地理分布**：跨地域的资源部署导致网络延迟差异
- **动态工作负载**：请求模式随时间变化的波动性
- **延迟要求差异**：不同应用对响应时间的敏感度不同
- **成本效率约束**：需要在性能和运营成本间取得平衡

传统的静态分配策略难以应对这些复杂、动态的环境需求。

## 技术方案

### 强化学习框架

#### 状态空间设计

- **节点配置**：CPU/GPU型号、内存大小、地理位置、节点网络速率
- **工作负载特征**：请求类型、输入长度、模型类型
- **系统状态**：队列长度、资源利用率、网络状况
- **时间因素**：周期性流量模式、请求时间分布

#### 动作空间设计

- **实例部署**：选择部署节点
- **资源分配**：CPU/GPU核心分配、内存分配
- **弹性伸缩**：实例数量动态调整
- **请求路由**：智能请求分配

#### 奖励函数设计

多目标优化函数，平衡以下因素：

```
奖励 = w1×吞吐量 + w2×(1/延迟) + w3×资源利用率 + w4×(1/成本) + w5×(1/错误率)
```

其中权重系数(w1-w5)可根据业务需求动态调整。

### RL算法对比

| 算法         | 类型     | 适用场景     | 优势                           |
| ------------ | -------- | ------------ | ------------------------------ |
| **策略梯度** | 策略优化 | 连续动作空间 | 直接优化策略，适合高维动作空间 |
| **A2C**      | 策略优化 | 并行环境     | 样本效率高，减少训练方差       |
| **DQN**      | 值优化   | 离散动作空间 | 稳定可靠，经验回放机制         |
| **QMIX**     | 多Agent  | 分布式决策   | 集中训练分散执行，适合集群环境 |

### RL模型建模

### 仿真环境

#### 数据集设计

数据集包含以下关键表结构：

**节点配置信息**
| 字段 | 描述 |
|------|------|
| 节点ID | 唯一标识符 |
| CPU使用率(%) | 当前CPU利用率 |
| 内存使用(MB) | 当前内存使用量 |
| GPU使用率(%) | 当前GPU利用率 |
| GPU内存使用(MB) | GPU显存使用量 |
| 磁盘IO(ops/s) | 磁盘操作频率 |
| 网络带宽(Mbps) | 可用网络带宽 |
| 节点位置 | 地理位置 |
| 实例数量 | 部署的实例数 |
| 实例类型 | 硬件配置类型 |

**请求特征**
| 字段 | 描述 |
|------|------|
| 请求ID | 唯一标识符 |
| 请求类型 | 文本生成/摘要生成/问答等 |
| 输入长度 | 输入token数量 |
| 输出长度 | 输出token数量 |
| 模型类型 | LLaMA-7B/GPT-3等 |
| 请求时间戳 | 请求到达时间 |
| 处理时间(ms) | 实际处理耗时 |
| 响应延迟(ms) | 端到端延迟 |
| 能耗(kWh) | 请求处理能耗 |
| 成本(元) | 请求处理成本 |
| 错误率(%) | 处理失败率 |

完整数据集模板见：[Datasets/simulation_dataset_template.csv](Datasets/simulation_dataset_template.csv)

要求：

- 模型类型固定为1-3个型号
- 节点配置中，CPU固定描述为多少cores，内存固定描述为多少GB，磁盘固定描述为多少GB，GPU型号固定为1-3个目前常用型号
-

## 核心功能

1. **异构资源管理**：优化不同硬件配置的资源分配
2. **地理感知调度**：根据节点位置智能分配请求，降低延迟
3. **动态弹性伸缩**：根据负载自动调整资源分配
4. **成本效率优化**：平衡性能与运营成本
5. **多目标优化**：同时优化多个可能冲突的目标
6. **容错机制**：自动处理节点故障和网络波动

## 数据集

数据集存储在`Datasets/`目录：

- `invoke_data2.csv`：工作负载模式和调用数据
- `load_test_results.csv`：负载测试性能指标，包含：
  - 资源利用率（CPU、内存、GPU）
  - 实例数量
  - 吞吐量（RPS）
  - 延迟指标（P50、P95、P99）
  - 时间戳
- `simulation_dataset_template.csv`：仿真数据集模板

## 评估指标

1. **平均响应时间**：所有请求的平均处理时间
2. **吞吐量**：每秒处理的请求数（RPS）
3. **资源利用率**：CPU、内存、GPU使用效率
4. **单次推理成本**：总运营成本 / 总请求数
5. **容错率**：系统在负载波动下的稳定性
6. **SLA达成率**：满足服务等级协议的比例

## 实现路线图

1. **基础框架搭建**（当前阶段）
   - 仿真环境开发
   - 数据集生成工具
   - 基本RL算法实现
2. **算法优化阶段**
   - 多Agent系统开发
   - 自适应奖励函数
   - 迁移学习支持

## 使用说明

### RL算法对比

| 算法         | 类型     | 适用场景     | 优势                           |
| ------------ | -------- | ------------ | ------------------------------ |
| **策略梯度** | 策略优化 | 连续动作空间 | 直接优化策略，适合高维动作空间 |
| **A2C**      | 策略优化 | 并行环境     | 样本效率高，减少训练方差       |
| **DQN**      | 值优化   | 离散动作空间 | 稳定可靠，经验回放机制         |
| **QMIX**     | 多Agent  | 分布式决策   | 集中训练分散执行，适合集群环境 |

---

## 📊 真实世界数据集设计规范

为了确保 RL 算法在接近生产环境的条件下进行训练与评估，本项目拒绝使用小规模玩具数据，而是**通过程序化生成引擎构建大规模、符合真实物理约束的仿真数据集**，并严格划分为训练集、验证集和测试集。

### 1. 硬件与模型约束（强制规范）

在生成数据时，必须严格遵循以下离散枚举值，禁止随机生成无意义的型号：

- **支持的模型类型（固定3种）**：
  1. `LLaMA3-7B` (轻量级，低延迟要求)
  2. `Qwen-14B` (中等规模，均衡型)
  3. `DeepSeek-70B` (重量级，高显存与长上下文要求)
- **支持的GPU型号（固定3种常用型号）**：
  1. `NVIDIA A100-SXM4-80GB` (高端算力)
  2. `NVIDIA L40S-48GB` (中端推理主力)
  3. `NVIDIA RTX 4090-24GB` (消费级/边缘算力)
- **资源描述规范**：
  - CPU：必须描述为核数（如 `64 cores`, `128 cores`）
  - 内存：必须描述为GB（如 `256 GB`, `512 GB`）
  - 磁盘：必须描述为GB（如 `2000 GB NVMe`）

### 2. 数据集构成与规模要求

生成的数据集存放在 `Datasets/generated/` 目录下，分为以下两类文件：

#### A. 静态集群拓扑数据集 (`cluster_profiles_*.csv`)

描述物理世界的异构节点池，不同划分对应不同的节点组合，防止策略过拟合单一拓扑。

- **训练集** (`cluster_profiles_train.csv`)：随机组合生成 **5,000 个**异构节点配置。
- **验证集** (`cluster_profiles_val.csv`)：随机组合生成 **1,000 个**异构节点配置。
- **测试集** (`cluster_profiles_test.csv`)：包含极端场景（如全A100集群、全4090集群、跨地域高延迟集群），生成 **500 个**节点配置。
- **核心字段**：`node_id, region(北京/上海/广州), cpu_cores, mem_gb, disk_gb, gpu_model, gpu_mem_gb, base_cost_per_hour, network_bandwidth_mbps`。

#### B. 动态工作负载数据集 (`workload_streams_*.csv`)

描述到达系统的请求时间序列，拟合真实的流量潮汐效应和长尾分布。

- **训练集** (`workload_streams_train.csv`)：模拟 30 天周期，包含 **> 2,000,000 条**请求记录。
- **验证集** (`workload_streams_val.csv`)：模拟 3 天周期，包含 **> 200,000 条**请求记录。
- **测试集** (`workload_streams_test.csv`)：模拟 3 天包含突发流量的周期，包含 **> 200,000 条**请求记录。
- **核心字段**：`timestamp, req_id, model_type(限填3种), input_tokens(符合长尾分布), output_tokens(符合长尾分布)`。

---

## 🗺️ 开发里程碑与详细需求说明书

> **说明**：严格按照以下 5 个 Phase 迭代。每个阶段明确了交付物、接口规范和验收标准。

### Phase 1: 基础设施与海量数据生成底座

**目标**：不写任何 RL 代码，纯 Python/PyTorch 构建高保真数据生成器与仿真环境骨架。

- **需求 1.1：集群拓扑数据生成器**
  - **文件**：`src/utils/data_generators/cluster_generator.py`
  - **类/方法**：`class ClusterTopologyGenerator`
  - **详细要求**：
    - 读取预定义的硬件字典（A100/L40S/4090的参数）。
    - 实现 `generate(num_nodes: int, split: str)` 方法。
    - 能够随机组合硬件，并根据地理位置（如北京->上海增加 20ms 基础延迟）计算网络拓扑矩阵。
    - 按比例输出 `train/val/test` 三个 CSV 文件，确保条数达到 5k/1k/500 的标准。
- **需求 1.2：工作负载流数据生成器**
  - **文件**：`src/utils/data_generators/workload_generator.py`
  - **类/方法**：`class WorkloadStreamGenerator`
  - **详细要求**：
    - 使用 `numpy` 拟合泊松过程模拟请求到达。
    - Token 长度必须使用对数正态分布或伽马分布模拟真实的长尾特征（大部分请求很短，少数极长）。
    - 实现潮汐效应：在早9点、晚8点注入正弦波流量高峰。
    - 输出 `train/val/test` 三个 CSV，确保条数达到 200万/20万/20万的标准。必须包含lazy-writing机制，避免 200 万条数据直接撑爆内存。
- **需求 1.3：数据加载与预处理工具**
  - **文件**：`src/utils/data_loader.py`
  - **类/方法**：`class DataPipeline`
  - **详细要求**：实现 `load_cluster(split)` 和 `get_workload_iterator(split)`。由于工作负载数据量大，迭代器需支持按批次从磁盘读取。
- **需求 1.4：Gymnasium 环境核心骨架**
  - **文件**：`src/envs/cluster_env.py`
  - **类/方法**：`class LLMClusterEnv(gym.Env)`
  - **详细要求**：
    - `__init__` 接收 `DataPipeline` 提供的初始状态，定义 `observation_space` 和 `action_space`。
    - `step(action)`：接收动作（节点ID），根据当前时刻的 Workload 特征，模拟资源扣减（基于选定的模型类型和GPU型号计算推理时间），计算奖励。
- **需求 1.5：独立奖励函数模块**
  - **文件**：`src/envs/reward_fn.py`
  - **类/方法**：`def calculate_reward(prev_state, current_state, action, config) -> float`
  - **详细要求**：实现多目标加权逻辑。若请求被分配到显存不足的节点（如将 GPT-3.5-16K 分配给 24GB 的 4090 且无显存优化），直接返回极大负惩罚。
- **🧪 验收标准**：成功运行生成脚本，在 `Datasets/generated/` 下生成符合规范的 6 个 CSV 文件（合计超 240 万行）。随机抽样检查：GPU型号必须属于指定的3种，Token数呈现长尾分布。

### Phase 2: 单 Agent 基线算法验证

**目标**：实现 DQN，跑通基于海量数据的训练闭环。

- **需求 2.1：算法基础抽象类** (`src/algorithms/base_agent.py`)：定义 `act`, `learn`, `save`, `load` 抽象接口。
- **需求 2.2：高效经验回放池** (`src/algorithms/replay_buffer.py`)：支持百万级 Transition 存储，支持 Prioritized Experience Replay (PER) 或标准均匀采样。
- **需求 2.3：DQN 算法实现** (`src/algorithms/dqn.py`)：包含 Double DQN 和 Dueling DQN 架构。输入为环境 State 张量，输出为各个节点的 Q 值。
- **需求 2.4：基础训练主循环** (`scripts/train_dqn.py`)：对接 Env 和 Agent，实现 Epsilon 衰减，每 1000 步保存一次 Checkpoint。
- **🧪 验收标准**：在包含 100 个节点的子集上运行 DQN，Reward 曲线在 5 万步内呈现上升趋势。

### Phase 3: 进阶单智能体与调参

**目标**：引入策略梯度，对比离散与连续动作空间。

- **需求 3.1：策略梯度 (REINFORCE)** (`src/algorithms/policy_gradient.py`)
- **需求 3.2：A2C 算法** (`src/algorithms/a2c.py`)：实现 Actor-Critic 双头网络与 GAE 优势估计。
- **需求 3.3：统一评估框架** (`src/utils/evaluation.py`)：关闭探索，运行完整测试集流，输出 `{平均延迟, P99延迟, 吞吐量, 总成本}` 汇总字典。
- **🧪 验收标准**：使用评估框架对比 DQN、PG、A2C 在相同测试集流上的表现，生成对比表格。

### Phase 4: 多 Agent 分布式决策扩展

**目标**：拆解单一调度器，每个节点作为一个 Agent，使用 QMIX 协作。

- **需求 4.1：多智能体环境包装器** (`src/envs/multi_cluster_wrapper.py`)：将全局 Obs 拆分为 $N$ 个局部 Obs，Action 也拆分为 $N$ 个。
- **需求 4.2：QMIX 组件** (`src/algorithms/qmix/components.py`)：实现 `RNNAgent` (处理时序请求流) 和满足单调性约束的 `QMixer` 网络。
- **需求 4.3：QMIX 训练器** (`src/algorithms/qmix/qmix_trainer.py`)：实现 CTDE（集中训练分散执行）逻辑。
- **🧪 验收标准**：在多节点环境下，验证边缘节点满载时能通过 QMIX 自发将流量路由至云端节点。

### Phase 5: 评估体系与工程化收尾

**目标**：输出科研成果级别的图表与严谨的测试。

- **需求 5.1：边界与压力测试** (`tests/test_env.py`)：使用极端流量（如瞬间 10倍峰值）测试环境是否崩溃，奖励计算是否出现 NaN。
- **需求 5.2：可视化模块** (`src/utils/visualization.py`)：使用 Plotly 绘制训练 Loss 曲线、奖励移动平均线；绘制不同算法的多维雷达图（延迟/成本/吞吐/利用率）。
- **🧪 验收标准**：`pytest tests/` 100% 通过，生成可直接用于论文的交互式 HTML 图表。

---

## 📁 项目结构示例

RLBigHW/
├── Datasets/
│ ├── invoke_data2.csv # 原始参考流量
│ ├── load_test_results.csv # 原始参考压测
│ ├── simulation_dataset_template.csv # 原始模板
│ └── generated/ # [新生成] 大规模仿真数据
│ ├── cluster_profiles_train.csv # 5000条节点配置
│ ├── cluster_profiles_val.csv # 1000条节点配置
│ ├── cluster_profiles_test.csv # 500条节点配置
│ ├── workload_streams_train.csv # >200万条请求流
│ ├── workload_streams_val.csv # >20万条请求流
│ └── workload_streams_test.csv # >20万条请求流
│
├── src/
│ ├── envs/
│ │ ├── **init**.py
│ │ ├── cluster_env.py # 主环境类
│ │ ├── reward_fn.py # 奖励函数
│ │ └── multi_cluster_wrapper.py # 多智能体包装器
│ ├── algorithms/
│ │ ├── **init**.py
│ │ ├── base_agent.py # 抽象基类
│ │ ├── replay_buffer.py # 回放池
│ │ ├── dqn.py # DQN
│ │ ├── policy_gradient.py # PG
│ │ ├── a2c.py # A2C
│ │ └── qmix/ # QMIX
│ ├── utils/
│ │ ├── data_loader.py # 数据加载管道
│ │ ├── data_generators/ # [核心] 数据生成引擎
│ │ │ ├── cluster_generator.py # 节点拓扑生成
│ │ │ └── workload_generator.py # 请求流生成
│ │ ├── metrics.py
│ │ └── visualization.py
│ └── configs/
│ ├── dqn_config.yaml
│ └── env_config.yaml
│
├── tests/
├── scripts/ # 执行脚本 (如 train_dqn.py)
├── Readme.md
├── CLAUDE.md
└── requirements.txt

## 📈 评估指标

我们拒绝单一指标，采用多维雷达图评估模型效能：

1. **经济效益**：单次推理成本 = $\frac{\sum (节点能耗成本 + 硬件折旧成本)}{总成功请求数}$
2. **时效性**：P99 Tail 延迟（排除长尾效应干扰）、平均响应时间。
3. **资源效率**：加权资源利用率 = $\frac{0.4 \times CPU_{util} + 0.4 \times GPU_{util} + 0.2 \times MEM_{util}}{总分配资源}$
4. **稳定性**：SLA 达成率（延迟小于 200ms 的请求占比）、节点过载发生率。

## 🤝 贡献指南

欢迎加入我们打造下一代 AI 基础设施调度系统！请阅读 [`CLAUDE.md`](CLAUDE.md) 了解代码规范。

1. Fork 本仓库。
2. 创建特性分支 (`git checkout -b feature/Algorithm-PPO`)。
3. 确保代码通过 `pytest tests/` 且覆盖率达到 80% 以上。
4. 提交 PR 并附上算法在仿真环境中的性能对比截图。

**注意**：所有主要更改应添加测试用例，并确保通过现有测试。

**注意**：随着项目进展，请及时更新此文件以反映实际代码结构和开发实践。

# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在本仓库工作的指导。

## 项目概述

本项目专注于使用强化学习（RL）优化大型语言模型（LLM）在异构基础设施上的资源分配。主要目标包括：

1. **基于 RL 的资源优化**：开发 RL 算法优化 LLM 实例在不同硬件配置和地理位置的部署
2. **单 Agent vs 多 Agent 系统**：比较集中式与分布式决策方法的性能
3. **RL 算法评估**：对比策略梯度、A2C、DQN、QMIX 等不同 RL 算法
4. **模拟数据集生成**：创建用于训练和评估的仿真数据集
5. **大规模集群管理**：设计适用于大规模集群管理的解决方案

## 关键考量

- **仿真工作**：需要开发完整的仿真环境用于数据集生成和算法测试
- **算法实现**：需要实现多种 RL 算法并进行公平对比
- **可扩展性**：解决方案需支持管理数百节点的大型集群
- **性能指标**：优化目标包括：
  - 资源利用率（CPU、内存、GPU）
  - 延迟（边缘、云、全局）
  - 成本效益（每次推理成本）
  - 容错率（故障率）
- **多目标优化**：需要平衡可能冲突的优化目标

## 仓库结构

```
RLBigHW/
├── Datasets/                  # 仿真数据集
│   ├── invoke_data2.csv       # 工作负载模式数据
│   ├── load_test_results.csv  # 负载测试性能指标
│   └── simulation_dataset_template.csv # 数据集模板
├── src/                       # 源代码目录
│   ├── envs/                  # RL 环境实现
│   ├── algorithms/            # RL 算法实现
│   └── utils/                 # 工具函数
├── tests/                     # 测试代码目录
├── scripts/                   # 执行脚本
├── Readme.md                  # 项目文档
└── CLAUDE.md                  # Claude 专用指南
```

---

## 开发指南

### 初始实现步骤

1. **RL 环境搭建**：
   - 定义状态空间（机器配置、工作负载、地理位置）
   - 定义动作空间（实例部署、资源分配、扩展决策）
   - 设计奖励函数（平衡延迟、吞吐量、资源利用率和成本）

2. **仿真能力开发**：
   - 实现工作负载生成器
   - 开发资源监控模块
   - 创建成本计算模型

3. **基准算法实现**：
   - 按优先级顺序实现：DQN → A2C → Policy Gradient → QMIX
   - 每种算法实现标准化接口

4. **评估框架**：
   - 创建统一评估指标
   - 开发结果可视化工具
   - 实现 A/B 测试框架

---

## 详细实现规范

### 状态空间编码规范

**状态向量结构**（总维度：~256）：

```python
# src/envs/state_encoder.py

class StateEncoder:
    """状态编码器：将原始观测转换为神经网络输入。"""
    
    def encode(self, node_states: dict, workload_queue: list, global_info: dict) -> torch.Tensor:
        """
        编码状态为固定维度的向量。
        
        输出结构：
        - [0:64]   节点资源状态（每个节点 8 维 × 最多 8 节点或统计特征）
        - [64:128] 工作负载统计特征（64 维）
        - [128:192] 地理位置编码（one-hot + 延迟矩阵展平）
        - [192:224] 时间特征编码（小时、星期的正弦/余弦编码）
        - [224:256] 全局统计（平均队列长度、总待处理请求数等）
        """
        # 实现细节...
        pass
```

**时间特征编码**（避免不连续性）：
```python
def encode_time(hour: int, day_of_week: int) -> np.ndarray:
    """将时间编码为连续的正弦/余弦特征。"""
    # 小时编码 (2 维)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # 星期编码 (2 维)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    return np.array([hour_sin, hour_cos, dow_sin, dow_cos])
```

### 动作空间设计详解

**离散动作空间**（DQN/QMIX）：
```python
# src/envs/action_space.py

class DiscreteActionSpace:
    """
    离散动作空间设计。
    
    动作编码方案：
    - action = node_id * num_resource_levels + resource_level
    
    示例：100 个节点 × 3 个资源等级 = 300 个离散动作
    """
    
    def __init__(self, num_nodes: int, num_resource_levels: int = 3):
        self.num_nodes = num_nodes
        self.num_resource_levels = num_resource_levels
        self.n = num_nodes * num_resource_levels
    
    def decode(self, action: int) -> Tuple[int, int]:
        """将动作索引解码为 (节点 ID, 资源等级)。"""
        node_id = action // self.num_resource_levels
        resource_level = action % self.num_resource_levels
        return node_id, resource_level
```

**资源等级定义**：
| 等级 | CPU 分配 | 内存分配 | GPU 分配 | 适用场景 |
|------|----------|----------|----------|----------|
| 低配 | 2 cores | 8 GB | 共享 | 小模型/低优先级 |
| 标准 | 4 cores | 16 GB | 独占 | 中等模型/标准优先级 |
| 高配 | 8 cores | 32 GB | 独占 + 预留 | 大模型/高优先级 |

### 奖励函数实现细节

```python
# src/envs/reward_fn.py

def calculate_reward(
    prev_state: dict,
    current_state: dict,
    action: int,
    info: dict,
    config: dict
) -> float:
    """
    计算多目标加权奖励。
    
    Args:
        prev_state: 上一时刻状态
        current_state: 当前时刻状态
        action: 执行的动作
        info: 包含详细信息的字典
        config: 奖励权重配置
        
    Returns:
        标量奖励值
        
    奖励组成：
    1. 基础奖励：吞吐量、延迟、利用率
    2. 约束惩罚：SLA 违规、资源溢出
    3. 稀疏奖励：episode 结束时的总评估
    """
    weights = config.get('reward_weights', {
        'throughput': 0.3,
        'latency': 0.3,
        'utilization': 0.2,
        'cost': 0.1,
        'sla': 0.1
    })
    
    # 1. 吞吐量奖励（已完成请求数）
    throughput_reward = info['completed_requests'] * 0.01
    
    # 2. 延迟惩罚（基于平均延迟与 SLA 的比率）
    avg_latency = info.get('avg_latency', 0)
    target_latency = info.get('target_latency', 200)
    latency_reward = -np.power(avg_latency / target_latency, 2)
    
    # 3. 资源利用率奖励（目标：60-80% 利用率）
    utilization = info.get('avg_utilization', 0)
    target_util = 0.7
    utilization_reward = 1.0 - abs(target_util - utilization)
    
    # 4. 成本惩罚
    cost = info.get('operational_cost', 0)
    revenue = info.get('revenue', 1)
    cost_reward = -(cost / revenue) if revenue > 0 else -1.0
    
    # 5. SLA 达成奖励
    sla_rate = info.get('sla_compliance_rate', 1.0)
    sla_reward = sla_rate
    
    # 加权求和
    total_reward = (
        weights['throughput'] * throughput_reward +
        weights['latency'] * latency_reward +
        weights['utilization'] * utilization_reward +
        weights['cost'] * cost_reward +
        weights['sla'] * sla_reward
    )
    
    # 约束惩罚
    if info.get('oom_occurred', False):
        total_reward -= 10.0
    if info.get('queue_overflow', False):
        total_reward -= 5.0
    if sla_rate < 0.95:
        total_reward -= 2.0
    
    return total_reward
```

### 数据集说明

- **invoke_data2.csv**：包含请求调用模式和时间分布
- **load_test_results.csv**：包含以下负载测试指标：
  - 资源利用率（CPU、内存、GPU）
  - 实例数量与类型
  - 吞吐量（RPS）
  - 延迟指标（P50、P95、P99）
  - 时间戳序列

### RL 算法对比维度

| 算法         | 适用场景     | 优势             | 劣势           |
| ------------ | ------------ | ---------------- | -------------- |
| **策略梯度** | 连续动作空间 | 直接优化策略     | 高方差、收敛慢 |
| **A2C**      | 并行环境     | 样本效率高       | 超参数敏感     |
| **DQN**      | 离散动作空间 | 稳定、可扩展     | 过估计问题     |
| **QMIX**     | 多 Agent 协作  | 集中训练分散执行 | 仅适用合作场景 |

### 数据集规范

1. **硬件配置**
   - CPU: 固定为 8 核/16 核/32 核/64 核
   - 内存：固定为 32GB/64GB/128GB/256GB
   - GPU: 固定为 NVIDIA A100(40GB)/NVIDIA L40(48GB)/NVIDIA RTX4090(24GB)

2. **工作负载特征**
   - 请求类型：文本生成/摘要生成/问答
   - 输入长度：50-4096 tokens (对数正态分布)
   - 模型类型：LLaMA-7B/GPT-3.5/Qwen-14B

3. **数据生成要求**（中等规模方案）
   - 集群拓扑训练集：500 个节点配置
   - 集群拓扑验证集：100 个节点配置
   - 集群拓扑测试集：50 个节点配置（含极端场景）
   - 工作负载训练集：~50,000 条请求（模拟 7 天）
   - 工作负载验证集：~10,000 条请求（模拟 1 天）
   - 工作负载测试集：~10,000 条请求（含突发流量）

### 实现路线图

1. **Phase 1: 基础环境搭建**
   - 实现 Gymnasium 兼容的 RL 环境
   - 开发数据集生成工具
   - 定义状态/动作/奖励函数

2. **Phase 2: DQN 实现**
   - 网络架构：3 层 MLP
   - 经验回放缓冲区
   - 目标网络实现

3. **Phase 3: 多算法扩展**
   - A2C 实现
   - 策略梯度实现
   - QMIX 实现

4. **Phase 4: 评估优化**
   - 多目标优化算法
   - 迁移学习支持
   - 大规模集群测试

### 评估指标

1. **性能指标**
   - 平均响应时间 (ms)
   - P99 延迟 (ms)
   - 吞吐量 (RPS)

2. **效率指标**
   - 资源利用率 (%)
   - 单次推理成本 (元)
   - 能效比 (RPS/W)

3. **稳定性指标**
   - SLA 达成率 (%)
   - 故障恢复时间 (s)
   - 过载发生率 (%)

---

## 开发规范

### 代码结构

- `src/envs/`: RL 环境实现
- `src/algorithms/`: RL 算法实现
- `src/utils/`: 工具函数
- `tests/`: 单元测试
- `scripts/`: 执行脚本
- `src/configs/`: 配置文件

### 编码标准

1. **语言规范**
   - Python 3.10+
   - 使用类型注解（Type Hints）
   - 遵循 PEP8 规范
   - 所有公共函数必须包含文档字符串（Google 风格）

2. **文件组织**
   - 每个模块应有清晰的单一职责
   - 避免循环导入
   - 使用 `__init__.py` 显式导出公共接口

3. **命名规范**
   - 类名：大驼峰（如 `LLMClusterEnv`）
   - 函数/变量：小写 + 下划线（如 `calculate_reward`）
   - 常量：全大写 + 下划线（如 `DEFAULT_BATCH_SIZE`）
   - 私有成员：单下划线前缀（如 `_internal_state`）

4. **文档字符串格式**（Google Style）
```python
def process_batch(data: List[Dict], batch_size: int) -> torch.Tensor:
    """处理批次数据并转换为张量。
    
    Args:
        data: 输入数据列表，每个元素为包含特征字典的样本
        batch_size: 批次大小，必须为正整数
        
    Returns:
        处理后的张量，shape 为 (batch_size, feature_dim)
        
    Raises:
        ValueError: 当 batch_size <= 0 时抛出
    """
```

### 测试策略

1. **测试层级**
   - **单元测试**：测试单个函数/类的功能（`tests/unit/`）
   - **集成测试**：测试模块间交互（`tests/integration/`）
   - **端到端测试**：测试完整训练/评估流程（`tests/e2e/`）

2. **测试覆盖率要求**
   - 核心算法模块：>90%
   - 环境模块：>85%
   - 工具函数：>80%
   - 整体覆盖率：>80%

3. **测试编写规范**
```python
import pytest
from src.envs.cluster_env import LLMClusterEnv

class TestLLMClusterEnv:
    """LLMClusterEnv 的单元测试类。"""
    
    def test_step_invalid_action_raises(self):
        """测试无效动作应抛出异常。"""
        env = LLMClusterEnv(num_nodes=10)
        env.reset()
        
        # 动作超出节点范围应抛出 ValueError
        with pytest.raises(ValueError, match="Invalid node id"):
            env.step(-1)  # 负数节点 ID
    
    def test_reset_returns_valid_observation(self):
        """测试 reset 返回的观测值格式正确。"""
        env = LLMClusterEnv(num_nodes=10)
        obs = env.reset()
        
        assert 'node_states' in obs
        assert obs['node_states'].shape[0] == 10
```

4. **运行测试**
```bash
# 运行所有测试
pytest tests/

# 运行特定模块测试
pytest tests/unit/test_env.py

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 日志规范

1. **日志级别使用**
   - `DEBUG`：详细调试信息，用于开发阶段
   - `INFO`：关键节点状态（训练开始、保存 checkpoint）
   - `WARNING`：可恢复的异常情况
   - `ERROR`：导致功能失败的错误
   - `CRITICAL`：系统级故障

2. **日志格式**
```python
import logging

logger = logging.getLogger(__name__)

# 推荐格式
logger.info(f"Episode {episode}: reward={reward:.2f}, avg_reward={avg_reward:.2f}")

# 避免在高频循环中输出 INFO/DEBUG 日志
```

3. **日志配置示例**
```yaml
# src/configs/logging.yaml
version: 1
handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
  file:
    class: logging.FileHandler
    filename: logs/training.log
    level: INFO
formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

### Git 版本控制策略

1. **分支模型**
```
main          - 生产就绪代码，保护分支
  ├── develop      - 集成分支，所有特性分支合并至此
  │     ├── feature/dqn-implementation
  │     ├── feature/a2c-algorithm
  │     └── fix/reward-calculation-bug
  └── release/v1.0 - 发布候选分支
```

2. **提交信息规范**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type 类型**：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档变更
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具/配置

**示例**：
```
feat(env): 添加 Gymnasium 兼容的环境骨架

- 实现 LLMClusterEnv 类
- 定义 observation_space 和 action_space
- 实现 step 和 reset 方法

Closes #12
```

3. **代码审查清单**
   - [ ] 代码通过所有测试
   - [ ] 覆盖率不低于 80%
   - [ ] 遵循编码规范
   - [ ] 添加了适当的日志
   - [ ] 更新了相关文档

### 错误处理规范

1. **异常处理原则**
   - 只捕获你能处理的异常
   - 使用具体的异常类型，避免裸 `except:`
   - 在边界层（API 入口、文件 IO）进行异常转换

2. **自定义异常**
```python
# src/utils/exceptions.py

class RLBigHWError(Exception):
    """基础异常类。"""
    pass

class InvalidActionError(RLBigHWError):
    """当动作不合法时抛出。"""
    pass

class ResourceExhaustedError(RLBigHWError):
    """当资源不足时抛出。"""
    pass

class InsufficientVRAMErrror(RLBigHWError):
    """当 GPU 显存不足时抛出。"""
    pass
```

### 配置管理

1. **配置文件格式**：使用 YAML
2. **配置加载**
```python
# src/utils/config.py
import yaml
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class EnvConfig:
    num_nodes: int = 100
    regions: List[str] = None
    gpu_types: List[str] = None
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = ["beijing", "shanghai", "guangzhou"]
        if self.gpu_types is None:
            self.gpu_types = ["A100-80GB", "L40S-48GB", "RTX4090-24GB"]

def load_config(path: str) -> EnvConfig:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return EnvConfig(**data)
```

3. **敏感信息管理**
   - 不要将密钥/凭证提交到仓库
   - 使用 `.env` 文件配合 `python-dotenv`
   - 在 `.gitignore` 中排除敏感文件

---

## 更新机制

- 每次添加新功能时更新此文档
- 重大架构变更时更新路线图
- 算法性能突破时更新评估指标
- 发现新的最佳实践时更新开发规范

随着项目进展，请及时更新此文件以反映实际代码结构和开发实践。

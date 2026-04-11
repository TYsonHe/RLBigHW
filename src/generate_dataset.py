import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 节点配置模板
EDGE_NODES = [
    {"type": "边缘", "cpu": "Intel i5-1135G7", "cores": 4, "memory": 8, "gpu": "NVIDIA Jetson Xavier", "gpu_count": 1},
    {"type": "边缘", "cpu": "Intel i7-1165G7", "cores": 4, "memory": 16, "gpu": "NVIDIA Jetson Orin", "gpu_count": 1}
]

CLOUD_NODES = [
    {"type": "云端", "cpu": "AMD EPYC 7763", "cores": 64, "memory": 256, "gpu": "NVIDIA A100", "gpu_count": 4},
    {"type": "云端", "cpu": "Intel Xeon Platinum 8380", "cores": 80, "memory": 512, "gpu": "NVIDIA H100", "gpu_count": 8}
]

# 地理位置配置
EDGE_LOCATIONS = ["上海", "北京", "广州", "成都", "武汉", "重庆", "苏州", "郑州", "合肥", "福州"]
CLOUD_LOCATIONS = ["上海", "北京", "深圳", "杭州", "天津", "南京", "西安", "长沙", "济南", "沈阳"]

# 请求类型配置
REQUEST_TYPES = ["文本生成", "问答", "摘要生成", "代码生成"]
MODEL_TYPES = ["LLaMA-7B", "LLaMA-13B", "GPT-3", "GPT-4"]


def generate_node_config(node_id, node_type):
    """生成节点配置信息"""
    if node_type == "边缘":
        config = random.choice(EDGE_NODES)
        location = random.choice(EDGE_LOCATIONS)
        # 边缘节点网络性能较差
        network_latency = random.uniform(20, 50)
        network_bandwidth = random.uniform(80, 150)
    else:
        config = random.choice(CLOUD_NODES)
        location = random.choice(CLOUD_LOCATIONS)
        # 云节点网络性能较好
        network_latency = random.uniform(10, 20)
        network_bandwidth = random.uniform(1000, 2000)

    return {
        "节点ID": node_id,
        "节点类型": config["type"],
        "CPU型号": config["cpu"],
        "CPU核心数": config["cores"],
        "内存(GB)": config["memory"],
        "GPU型号": config["gpu"],
        "GPU数量": config["gpu_count"],
        "地理位置": location,
        "CPU使用率(%)": random.uniform(30, 80),
        "内存使用率(%)": random.uniform(25, 70),
        "GPU使用率(%)": random.uniform(20, 85),
        "网络延迟(ms)": network_latency,
        "网络带宽(Mbps)": network_bandwidth,
        "能耗(kWh)": random.uniform(0.3, 0.6) if node_type == "边缘" else random.uniform(3.5, 6.0),
        "成本(元/小时)": random.uniform(0.1, 0.2) if node_type == "边缘" else random.uniform(1.3, 2.3)
    }


def generate_request_data(node_config):
    """生成请求数据"""
    request_type = random.choice(REQUEST_TYPES)
    model_type = random.choice(MODEL_TYPES)

    # 根据请求类型和模型类型确定输入输出长度范围
    if request_type == "文本生成":
        input_len = random.randint(256, 512)
        output_len = random.randint(64, 128)
    elif request_type == "问答":
        input_len = random.randint(128, 256)
        output_len = random.randint(32, 64)
    elif request_type == "摘要生成":
        input_len = random.randint(512, 1024)
        output_len = random.randint(128, 256)
    else:  # 代码生成
        input_len = random.randint(1024, 2048)
        output_len = random.randint(256, 512)

    # 处理时间与节点类型和模型复杂度相关
    base_time = input_len * 0.1 + output_len * 0.05
    if "LLaMA-13B" in model_type or "GPT-4" in model_type:
        base_time *= 1.5
    if node_config["节点类型"] == "边缘":
        base_time *= 1.8

    processing_time = base_time * random.uniform(0.9, 1.1)
    response_latency = processing_time + node_config["网络延迟(ms)"] * random.uniform(0.8, 1.2)

    # 错误率与节点负载相关
    error_rate = max(0.1, min(0.8, (node_config["CPU使用率(%)"] + node_config["GPU使用率(%)"]) / 200))

    return {
        "请求类型": request_type,
        "输入长度": input_len,
        "输出长度": output_len,
        "模型类型": model_type,
        "处理时间(ms)": processing_time,
        "响应延迟(ms)": response_latency,
        "错误率(%)": error_rate
    }


def generate_dataset(num_nodes, num_requests_per_node, output_path):
    """生成完整数据集"""
    data = []

    # 生成节点配置
    nodes = []
    for i in range(num_nodes):
        node_type = "边缘" if i < num_nodes // 2 else "云端"
        node_config = generate_node_config(i + 1, node_type)
        nodes.append(node_config)

    # 生成请求数据
    for node in nodes:
        base_timestamp = datetime(2025, 4, 16, 12, 0, 0)
        for j in range(num_requests_per_node):
            request_data = generate_request_data(node)

            # 合并节点配置和请求数据
            record = {**node, **request_data}
            record["时间戳"] = (base_timestamp + timedelta(seconds=j * 5)).isoformat()

            data.append(record)

    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"数据集已生成，保存至 {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成强化学习仿真数据集")
    parser.add_argument("--num-nodes", type=int, default=10, help="节点数量")
    parser.add_argument("--requests-per-node", type=int, default=10, help="每个节点的请求数量")
    parser.add_argument("--output", type=str, default="Datasets/generated_dataset.csv", help="输出文件路径")

    args = parser.parse_args()

    generate_dataset(args.num_nodes, args.requests_per_node, args.output)
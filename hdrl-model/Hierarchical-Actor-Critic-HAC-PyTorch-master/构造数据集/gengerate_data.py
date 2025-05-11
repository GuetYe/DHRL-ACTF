import json
import os

# 定义文件路径
input_file = r"D:\rerouting-drl\hdrl-model\Hierarchical-Actor-Critic-HAC-PyTorch-master\dataSet\weight\200-2023-12-10-16-05-28.txt"

# --------------------- 新增配置区域 ---------------------
# 关键链路定义（根据实际拓扑指定核心链路）
CORE_LINKS = [
    (4, 3), (8, 11), (15, 18), (16, 17), (22, 25)
]

# 跨域链路定义（假设域划分：域A=1-10，域B=11-20，域C=21-30）
CROSS_DOMAIN_LINKS = [
    (8, 11), (10, 16), (15, 19), (22, 25)
]

# 网络枢纽节点（用于稀疏连通性场景）
CRITICAL_HUBS = [15, 8, 22]

# ------------------------------------------------------

def read_file(file_path):
    """读取原始文件内容"""
    with open(file_path, "r") as f:
        lines = f.readlines()
    return lines

def parse_data(lines):
    """解析原始数据为结构化列表"""
    data = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                link_data = eval(line)  # 注意：实际应用建议用更安全的解析方式
                data.append(link_data)
            except Exception as e:
                print(f"Error parsing line: {line}\n{str(e)}")
    return data

def modify_data(data, scenario):
    """定向修改数据（按场景需求）"""
    modified_data = []
    
    for item in data:
        src, dst, metrics = item
        new_metrics = metrics.copy()  # 创建副本避免修改原始数据

        # ---------- 场景1: 正常流量（不修改） ----------
        if scenario == "normal":
            pass

        # ---------- 场景2: 突发流量（仅影响核心链路） ----------
        elif scenario == "burst":
            if (src, dst) in CORE_LINKS:
                # 注入突发流量：占用90%带宽，增加延迟和丢包
                new_metrics['used_bw'] += 0.9 * (new_metrics['free_bw'] + new_metrics['used_bw'])
                new_metrics['free_bw'] = 0.1 * (new_metrics['free_bw'] + new_metrics['used_bw'])
                new_metrics['delay'] *= 5  # 延迟增加5倍
                new_metrics['pkt_drop'] = int(new_metrics['used_bw'] * 0.05)  # 丢包率5%

        # ---------- 场景3: 链路故障（仅断开核心链路） ----------
        elif scenario == "fault":
            if (src, dst) in CORE_LINKS[:3]:  # 断开前3条核心链路
                new_metrics.update({
                    'free_bw': 0.0,
                    'used_bw': 0.0,
                    'delay': 999.0,
                    'loss': 1.0,
                    'pkt_drop': 100
                })

        # ---------- 场景4: 跨域路由（限制域间带宽） ----------
        elif scenario == "cross_domain":
            if (src, dst) in CROSS_DOMAIN_LINKS:
                # 域间带宽限制为原来的30%
                new_metrics['free_bw'] *= 0.3
                new_metrics['used_bw'] *= 0.3  # 假设总带宽不变
                new_metrics['delay'] *= 2  # 域间延迟加倍

        # ---------- 场景5: 高负载（全局负载提升） ----------
        elif scenario == "high_load":
            total_bw = new_metrics['free_bw'] + new_metrics['used_bw']
            new_metrics['used_bw'] = total_bw * 0.9  # 占用90%带宽
            new_metrics['free_bw'] = total_bw * 0.1
            # 根据负载动态调整延迟
            new_metrics['delay'] *= 1 + (new_metrics['used_bw'] / total_bw) * 5

        # ---------- 场景6: 稀疏连通性（断开枢纽节点） ----------
        elif scenario == "sparse":
            if src in CRITICAL_HUBS or dst in CRITICAL_HUBS:
                new_metrics.update({
                    'free_bw': 0.0,
                    'used_bw': 0.0,
                    'delay': 999.0,
                    'loss': 1.0
                })

        modified_data.append((src, dst, new_metrics))
    
    return modified_data

def save_data(data, output_file):
    """保存修改后的数据"""
    with open(output_file, "w") as f:
        for item in data:
            f.write(f"{item}\n")

def main():
    # 读取并解析原始数据
    lines = read_file(input_file)
    original_data = parse_data(lines)

    # 定义所有场景
    scenarios = {
        "normal": "normal.txt",
        "burst": "burst_traffic.txt",
        "fault": "link_failure.txt",
        "cross_domain": "cross_domain.txt",
        "high_load": "high_load.txt",
        "sparse": "sparse_connectivity.txt"
    }

    # 生成各场景数据
    for scenario, filename in scenarios.items():
        print(f"Generating {scenario} scenario...")
        modified = modify_data(original_data, scenario)
        save_data(modified, os.path.join("modified_data", filename))
        print(f"Saved: {filename}")

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("modified_data", exist_ok=True)
    main()
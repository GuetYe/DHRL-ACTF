# # -*- coding: utf-8 -*-
# """
# @File     : train.py
# @Date     : 2023-12-30
# @Author   : Terry_Li     --落花有意，流水无情。
# IDE       : PyCharm
# @Mail     : terry.ljq.dev@foxmail.com
# """

from transformer_network import Transformer
import torch
import dataSet
import config
from gpu import gpu
from data1 import *
from utils import read_pickle_and_modify, normalize_value
import torch.nn as nn

criterion = nn.MSELoss()

def test(model_path='./model_weight/10_transformer_model.pth', test_seq_length=5):
    """
    测试训练好的Transformer模型
    :param model_path: 训练好的模型权重路径
    :param test_seq_length: 测试序列长度（需与训练时一致）
    """
    # 1. 初始化模型（结构需与训练时完全一致）
    model = Transformer(n_encoder_inputs=1, n_decoder_inputs=1)
    
    # 2. 加载训练好的权重
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"成功加载模型权重：{model_path}")
    except FileNotFoundError:
        print(f"错误：模型文件 {model_path} 不存在！")
        return
    except RuntimeError as e:
        print(f"模型加载失败，请检查模型结构是否匹配：{str(e)}")
        return
    
    model.eval()  # 设置为评估模式
    model.to(gpu())  # 移动到GPU（如果可用）
    
    # 3. 准备测试数据加载器（建议使用独立测试集）
    # 示例使用相同数据路径，实际应使用独立测试数据
    test_loader = read_file_to_datasets(
        config.DATASET_PATH, 
        seq_lenth=5,
        ista=300,  # 假设前250个样本用于训练
        iend=600      # 后50个样本用于测试
    )
    
    # 4. 初始化评估指标
    total_loss = 0.0
    predictions = []
    ground_truth = []
    
    # 5. 禁用梯度计算
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # 生成解码器初始输入（与训练时一致）
            target_in = torch.zeros_like(inputs[:, :1, :])  # [batch, 1, features]
            
            # 前向传播
            outputs = model(inputs.to(gpu()), target_in.to(gpu()))
            
            # 计算损失
            loss = criterion(outputs, targets.to(gpu()))
            total_loss += loss.item()
            
            # 收集结果用于后续分析
            predictions.append(outputs.cpu().numpy())
            ground_truth.append(targets.numpy())
    
    # 6. 计算统计指标
    avg_loss = total_loss / len(test_loader)
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    
    # 7. 输出评估结果
    print("\n===== 测试结果 =====")
    print(f"平均MSE损失: {avg_loss:.4f}")
    print(f"样本数量: {len(ground_truth)}")
    
    # 8. 可视化示例预测（可选）
    plot_predictions(
        predictions[:10],  # 显示前5个样本
        ground_truth[:10],
        save_path="prediction_examples.png"
    )

def plot_predictions(preds, truths, save_path):
    """可视化预测结果"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    for i in range(min(3, preds.shape[1])):  # 最多显示3个特征
        plt.subplot(3, 1, i+1)
        plt.plot(truths[:, i], label='Ground Truth')
        plt.plot(preds[:, i], linestyle='--', label='Prediction')
        plt.title(f"Feature {i+1}")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"预测结果可视化已保存至：{save_path}")

if __name__ == "__main__":
    # 使用示例：测试第10个epoch保存的模型
    test(model_path='./model_weight/1_transformer_model.pth')
# DonutWhisper 多模态融合策略

本项目为 DonutWhisper 模型提供了多种多模态融合策略，用于改进音频和图像特征的融合效果。

## 融合策略概览

### 1. 线性融合 (Linear Fusion)
```python
fusion_type = "linear"
```
- **描述**: 简单的线性层融合
- **优点**: 计算简单，参数量少，训练稳定
- **缺点**: 表达能力有限，无法建模复杂交互
- **适用场景**: 快速原型验证，计算资源有限

### 2. 自注意力融合 (Self-Attention Fusion)
```python
fusion_type = "attention"
fusion_config = {"num_heads": 8, "dropout": 0.1}
```
- **描述**: 使用自注意力机制融合多模态特征
- **优点**: 能建模序列内交互，可学习注意力权重
- **缺点**: 计算复杂度高，需要更多训练数据
- **适用场景**: 需要建模序列内复杂关系

### 3. 交叉注意力融合 (Cross-Attention Fusion)
```python
fusion_type = "cross_attention"
fusion_config = {"num_heads": 8, "dropout": 0.1}
```
- **描述**: 音频和图像特征通过交叉注意力直接交互
- **优点**: 音频和图像直接交互，能学习模态间关系
- **缺点**: 需要对齐的模态数据，训练较复杂
- **适用场景**: 需要音频和图像深度交互

### 4. Q-Former 风格融合 (Q-Former Style Fusion)
```python
fusion_type = "q_former"
fusion_config = {"num_heads": 8, "dropout": 0.1}
```
- **描述**: 使用查询向量与多模态特征交互，类似 BLIP-2
- **优点**: 使用查询向量，灵活的特征提取，类似BLIP-2
- **缺点**: 参数量大，训练复杂
- **适用场景**: 需要灵活的特征提取和查询机制

## 使用方法

### 基本使用

```python
from model.donut_whisper import DonutWhisper

# 创建带有特定融合策略的模型
model = DonutWhisper(
    whisper_model=whisper_model,
    image_model_path=image_model_path,
    fusion_type="attention",  # 选择融合策略
    fusion_config={"num_heads": 8, "dropout": 0.1}  # 配置参数
)
```

### 训练脚本

```bash
# 使用注意力融合训练
python train_with_fusion.py \
    --fusion_type attention \
    --whisper_path /path/to/whisper \
    --image_model_path /path/to/donut \
    --data_path /path/to/data.json \
    --output_dir ./output

# 比较所有融合策略
python train_with_fusion.py --compare
```

### 测试融合层

```python
from model.fusion_examples import test_fusion_layer

# 测试所有融合策略
test_fusion_layer()
```

## 性能比较

| 融合策略 | 参数量 | 计算复杂度 | 训练稳定性 | 表达能力 |
|---------|--------|-----------|-----------|----------|
| Linear | 低 | 低 | 高 | 低 |
| Attention | 中 | 中 | 中 | 中 |
| Cross-Attention | 中 | 中 | 中 | 高 |
| Q-Former | 高 | 高 | 低 | 高 |

## 选择建议

### 根据数据规模选择
- **小数据集 (< 10K 样本)**: 推荐使用 `linear`
- **中等数据集 (10K - 100K 样本)**: 推荐使用 `attention` 或 `cross_attention`
- **大数据集 (> 100K 样本)**: 推荐使用 `q_former`

### 根据计算资源选择
- **计算资源有限**: 推荐使用 `linear`
- **计算资源充足**: 推荐使用 `q_former`

### 根据任务需求选择
- **需要音频图像深度交互**: 推荐使用 `cross_attention`
- **需要灵活查询**: 推荐使用 `q_former`

## 实验建议

1. **从简单开始**: 建议先尝试 `linear` 作为基线
2. **逐步复杂化**: 根据性能表现逐步尝试更复杂的融合策略
3. **超参数调优**: 对于每种融合策略，都需要调优相应的超参数
4. **消融实验**: 建议进行消融实验来验证每种融合策略的效果

## 注意事项

1. **数据对齐**: 某些融合策略（如 `cross_attention`）需要音频和图像特征在时间维度上对齐
2. **内存使用**: 复杂的融合策略会增加内存使用量，需要根据硬件调整批次大小
3. **训练稳定性**: 复杂的融合策略可能需要更仔细的学习率调度和正则化
4. **预训练权重**: 建议从预训练的 Whisper 和 Donut 模型开始，然后微调融合层

## 扩展建议

1. **混合融合**: 可以组合多种融合策略
2. **层次融合**: 在不同层次使用不同的融合策略
3. **动态融合**: 根据输入内容动态选择融合策略
4. **多尺度融合**: 在不同尺度上进行特征融合 

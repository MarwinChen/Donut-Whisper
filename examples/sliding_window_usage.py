"""
滑窗Q-former使用示例
展示如何使用滑窗Q-former进行多模态融合
"""

import torch
from model.donut_whisper import DonutWhisper
from model.sliding_window_config import (
    get_sliding_window_config,
    create_custom_sliding_window
)
from transformers import WhisperModel

def example_basic_usage():
    """基础使用示例"""
    print("=== 基础滑窗Q-former使用示例 ===")
    
    # 加载模型
    whisper_path = "/path/to/whisper/model"
    image_model_path = "/path/to/donut/model"
    
    whisper_model = WhisperModel.from_pretrained(whisper_path)
    
    # 使用预定义的滑窗配置
    config = get_sliding_window_config("basic")
    model = DonutWhisper(
        whisper_model=whisper_model,
        image_model_path=image_model_path,
        fusion_type=config["fusion_type"],
        fusion_config=config["fusion_config"]
    )
    
    print(f"模型融合类型: {config['fusion_type']}")
    print(f"窗口大小: {config['fusion_config']['window_size']}")
    print(f"滑窗步长: {config['fusion_config']['stride']}")
    print(f"是否重叠: {config['fusion_config']['overlap']}")
    
    return model

def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义滑窗配置示例 ===")
    
    # 创建自定义配置
    custom_config = create_custom_sliding_window(
        window_size=96,    # 自定义窗口大小
        stride=48,          # 自定义步长
        overlap=True,       # 保持重叠
        num_heads=12,       # 自定义注意力头数
        dropout=0.05        # 自定义dropout
    )
    
    print(f"自定义配置:")
    print(f"  窗口大小: {custom_config['fusion_config']['window_size']}")
    print(f"  滑窗步长: {custom_config['fusion_config']['stride']}")
    print(f"  是否重叠: {custom_config['fusion_config']['overlap']}")
    print(f"  注意力头数: {custom_config['fusion_config']['num_heads']}")
    print(f"  Dropout: {custom_config['fusion_config']['dropout']}")
    
    return custom_config

def example_different_scenarios():
    """不同场景的配置示例"""
    print("\n=== 不同场景配置示例 ===")
    
    scenarios = [
        ("基础场景", "basic"),
        ("长音频处理", "long_audio"),
        ("短音频处理", "short_audio"),
        ("高精度需求", "high_precision"),
        ("快速推理", "fast_inference")
    ]
    
    for scenario_name, config_name in scenarios:
        config = get_sliding_window_config(config_name)
        fusion_config = config["fusion_config"]
        
        print(f"\n{scenario_name}:")
        print(f"  窗口大小: {fusion_config['window_size']}")
        print(f"  滑窗步长: {fusion_config['stride']}")
        print(f"  是否重叠: {fusion_config['overlap']}")
        print(f"  注意力头数: {fusion_config['num_heads']}")
        print(f"  Dropout: {fusion_config['dropout']}")

def example_model_forward():
    """模型前向传播示例"""
    print("\n=== 模型前向传播示例 ===")
    
    # 模拟输入数据
    batch_size = 2
    audio_seq_len = 512
    image_seq_len = 256
    hidden_size = 768
    
    # 创建模拟数据
    audio_features = torch.randn(batch_size, audio_seq_len, hidden_size)
    image_features = torch.randn(batch_size, image_seq_len, hidden_size)
    audio_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool)
    image_mask = torch.ones(batch_size, image_seq_len, dtype=torch.bool)
    
    print(f"输入数据形状:")
    print(f"  音频特征: {audio_features.shape}")
    print(f"  图像特征: {image_features.shape}")
    print(f"  音频掩码: {audio_mask.shape}")
    print(f"  图像掩码: {image_mask.shape}")
    
    # 注意：这里只是示例，实际使用时需要加载真实的模型
    print("\n注意：这是示例代码，实际使用时需要加载真实的模型")

def main():
    """主函数"""
    print("滑窗Q-former使用示例")
    print("=" * 50)
    
    # 基础使用示例
    example_basic_usage()
    
    # 自定义配置示例
    example_custom_config()
    
    # 不同场景配置示例
    example_different_scenarios()
    
    # 模型前向传播示例
    example_model_forward()
    
    print("\n" + "=" * 50)
    print("示例完成！")
    print("\n使用建议:")
    print("1. 对于长音频，使用 'long_audio' 配置")
    print("2. 对于短音频，使用 'short_audio' 配置")
    print("3. 对于高精度需求，使用 'high_precision' 配置")
    print("4. 对于实时推理，使用 'fast_inference' 配置")
    print("5. 可以根据具体需求使用 create_custom_sliding_window() 创建自定义配置")

if __name__ == "__main__":
    main()











































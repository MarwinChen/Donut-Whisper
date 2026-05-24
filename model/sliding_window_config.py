"""
滑窗Q-former配置文件
提供不同场景下的滑窗参数配置
"""

# 基础滑窗配置
BASIC_SLIDING_WINDOW = {
    "fusion_type": "sliding_window_q_former",
    "fusion_config": {
        "window_size": 128,    # 音频窗口大小
        "stride": 64,          # 滑窗步长
        "overlap": True,       # 是否重叠
        "num_heads": 8,        # 注意力头数
        "dropout": 0.1         # dropout率
    }
}

# 长音频滑窗配置（适合处理长音频）
LONG_AUDIO_SLIDING_WINDOW = {
    "fusion_type": "sliding_window_q_former",
    "fusion_config": {
        "window_size": 256,    # 更大的窗口
        "stride": 128,         # 更大的步长
        "overlap": True,       # 保持重叠
        "num_heads": 16,       # 更多注意力头
        "dropout": 0.1
    }
}

# 短音频滑窗配置（适合处理短音频）
SHORT_AUDIO_SLIDING_WINDOW = {
    "fusion_type": "sliding_window_q_former",
    "fusion_config": {
        "window_size": 64,     # 较小的窗口
        "stride": 32,          # 较小的步长
        "overlap": True,       # 保持重叠
        "num_heads": 8,        # 标准注意力头数
        "dropout": 0.1
    }
}

# 高精度滑窗配置（适合需要高精度的场景）
HIGH_PRECISION_SLIDING_WINDOW = {
    "fusion_type": "sliding_window_q_former",
    "fusion_config": {
        "window_size": 96,     # 中等窗口大小
        "stride": 32,          # 小步长，高重叠
        "overlap": True,       # 保持重叠
        "num_heads": 12,       # 较多注意力头
        "dropout": 0.05        # 较低的dropout
    }
}

# 快速推理滑窗配置（适合实时推理）
FAST_INFERENCE_SLIDING_WINDOW = {
    "fusion_type": "sliding_window_q_former",
    "fusion_config": {
        "window_size": 192,    # 较大窗口，减少窗口数量
        "stride": 128,         # 大步长，减少计算
        "overlap": False,      # 不重叠，减少计算
        "num_heads": 8,        # 标准注意力头数
        "dropout": 0.1
    }
}

# 自定义滑窗配置函数
def create_custom_sliding_window(
    window_size=128,
    stride=64,
    overlap=True,
    num_heads=8,
    dropout=0.1
):
    """
    创建自定义滑窗配置
    
    Args:
        window_size (int): 音频窗口大小
        stride (int): 滑窗步长
        overlap (bool): 是否重叠
        num_heads (int): 注意力头数
        dropout (float): dropout率
    
    Returns:
        dict: 滑窗配置字典
    """
    return {
        "fusion_type": "sliding_window_q_former",
        "fusion_config": {
            "window_size": window_size,
            "stride": stride,
            "overlap": overlap,
            "num_heads": num_heads,
            "dropout": dropout
        }
    }

# 配置选择器
def get_sliding_window_config(config_name="basic"):
    """
    根据名称获取预定义的滑窗配置
    
    Args:
        config_name (str): 配置名称
    
    Returns:
        dict: 对应的滑窗配置
    """
    configs = {
        "basic": BASIC_SLIDING_WINDOW,
        "long_audio": LONG_AUDIO_SLIDING_WINDOW,
        "short_audio": SHORT_AUDIO_SLIDING_WINDOW,
        "high_precision": HIGH_PRECISION_SLIDING_WINDOW,
        "fast_inference": FAST_INFERENCE_SLIDING_WINDOW
    }
    
    if config_name not in configs:
        print(f"警告：未知的配置名称 '{config_name}'，使用基础配置")
        return BASIC_SLIDING_WINDOW
    
    return configs[config_name]











































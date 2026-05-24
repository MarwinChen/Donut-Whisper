"""
时序记录器，用于记录模型各模块的执行时间
"""
import time
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional
import json
import os


class TimingTracer:
    """时序记录器，用于记录模型各模块的执行时间"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings = defaultdict(list)  # 存储每个模块的时间记录
        self.current_trace = []  # 当前批次的时间轨迹
        self.hooks = []  # 存储注册的钩子
        
    def start_trace(self):
        """开始新的轨迹记录"""
        if self.enabled:
            self.current_trace = []
    
    def record(self, module_name: str, duration: float, metadata: Optional[Dict] = None):
        """记录模块的执行时间"""
        if self.enabled:
            record = {
                "module": module_name,
                "duration": duration,
                "metadata": metadata or {}
            }
            self.current_trace.append(record)
            self.timings[module_name].append(duration)
    
    def end_trace(self) -> List[Dict]:
        """结束当前轨迹记录并返回"""
        if self.enabled:
            trace = self.current_trace.copy()
            self.current_trace = []
            return trace
        return []
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.enabled:
            return {}
        
        stats = {}
        for module_name, durations in self.timings.items():
            if durations:
                stats[module_name] = {
                    "count": len(durations),
                    "total": sum(durations),
                    "mean": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "std": self._calculate_std(durations)
                }
        return stats
    
    @staticmethod
    def compute_statistics_from_traces(traces: List[List[Dict]]) -> Dict:
        """从轨迹列表中计算统计信息"""
        # 收集所有模块的时间
        module_times = defaultdict(list)
        
        for trace in traces:
            for record in trace:
                module_name = record.get("module", "unknown")
                duration = record.get("duration", 0.0)
                module_times[module_name].append(duration)
        
        # 计算统计信息
        stats = {}
        for module_name, durations in module_times.items():
            if durations:
                mean = sum(durations) / len(durations)
                stats[module_name] = {
                    "count": len(durations),
                    "total": sum(durations),
                    "mean": mean,
                    "min": min(durations),
                    "max": max(durations),
                    "std": TimingTracer._calculate_std(durations)
                }
        return stats
    
    @staticmethod
    def _calculate_std(values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def register_hooks(self, model: nn.Module):
        """为模型注册前向传播钩子"""
        if not self.enabled:
            return
        
        def make_hook(name):
            def forward_hook(module, input, output):
                # 注意：这个钩子只能记录模块被调用的时间点，不能直接测量执行时间
                # 实际的时序测量需要在代码中手动添加
                pass
            return forward_hook
        
        # 为关键模块注册钩子
        for name, module in model.named_modules():
            if any(keyword in name.lower() for keyword in ['encoder', 'decoder', 'fusion', 'linear']):
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def save_trace(self, filepath: str, traces: List[List[Dict]], statistics: Optional[Dict] = None):
        """保存轨迹到JSON文件"""
        if not self.enabled:
            return
        
        output = {
            "traces": traces,
            "statistics": statistics or self.get_statistics(),
            "summary": {
                "total_traces": len(traces),
                "total_modules": len(self.timings)
            }
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def reset(self):
        """重置所有记录"""
        self.timings.clear()
        self.current_trace = []
        self.remove_hooks()


class TimingContext:
    """上下文管理器，用于自动记录代码块的执行时间"""
    
    def __init__(self, tracer: TimingTracer, module_name: str, metadata: Optional[Dict] = None):
        self.tracer = tracer
        self.module_name = module_name
        self.metadata = metadata
        self.start_time = None
    
    def __enter__(self):
        if self.tracer.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracer.enabled and self.start_time is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            duration = time.time() - self.start_time
            self.tracer.record(self.module_name, duration, self.metadata)


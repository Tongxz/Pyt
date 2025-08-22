#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查GPU状态
"""

import torch
import os

print("=== GPU状态检查 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"  显存已用: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
else:
    print("CUDA不可用，将使用CPU")

print("\n=== MediaPipe GPU设置 ===")
print(f"MEDIAPIPE_DISABLE_GPU环境变量: {os.environ.get('MEDIAPIPE_DISABLE_GPU', '未设置')}")

# 测试MediaPipe GPU自动配置
try:
    import sys
    sys.path.insert(0, '.')
    from src.core.pose_detector import _gpu_enabled
    mediapipe_mode = "GPU模式" if _gpu_enabled else "CPU模式"
    print(f"MediaPipe自动配置结果: {mediapipe_mode}")
except Exception as e:
    print(f"无法获取MediaPipe配置状态: {e}")
    mediapipe_mode = "未知"

print("\n=== 当前使用的检测器 ===")
print("1. YOLO人体检测器 (YOLOv8) - 支持GPU加速")
print(f"2. MediaPipe姿态检测器 ({mediapipe_mode}) - 智能GPU检测")
print("3. 增强手部检测器 (集成多种检测方法)")
print("4. 运动分析器 (基于关键点轨迹)")
print("5. 行为识别器 (基于规则和特征)")
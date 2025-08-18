# MediaPipe 集成说明

## 概述

本项目已成功将 MediaPipe 集成到现有的行为识别系统中，用于增强洗手检测的准确性和可靠性。

## 主要功能

### 1. MediaPipe 手部检测增强
- 利用 MediaPipe 的高精度手部关键点检测
- 提供21个手部关键点的精确位置信息
- 增强现有手部区域检测的准确性

### 2. 增强的洗手行为识别
- 结合传统计算机视觉方法和 MediaPipe 技术
- 更准确的手指展开程度分析
- 改进的手部运动模式识别
- 提高洗手行为检测的置信度

## 技术实现

### 集成方式
- 直接集成到 `src/core/behavior.py` 中的 `BehaviorRecognizer` 类
- 保持向后兼容性，可选择启用/禁用 MediaPipe 功能
- 优雅降级：当 MediaPipe 不可用时自动回退到基础检测方法

### 核心方法
1. `_enhance_hand_detection_with_mediapipe()` - MediaPipe 手部检测增强
2. `_analyze_mediapipe_finger_spread()` - 基于关键点的手指展开分析
3. 增强的 `_analyze_hand_motion()` - 结合 MediaPipe 数据的运动分析

## 使用方法

### 初始化
```python
from src.core.behavior import BehaviorRecognizer

# 启用 MediaPipe 增强功能
behavior_recognizer = BehaviorRecognizer(
    confidence_threshold=0.3,
    use_mediapipe=True  # 启用 MediaPipe
)
```

### 洗手检测
```python
# 进行洗手检测
confidence = behavior_recognizer.detect_handwashing(
    person_bbox=person_bbox,
    hand_regions=hand_regions,
    frame=image_frame
)
```

## 测试验证

运行测试脚本验证 MediaPipe 集成功能：
```bash
python test_mediapipe_integration.py
```

测试脚本会：
- 使用 `tests/fixtures` 目录下的真实测试数据
- 验证 MediaPipe 手部检测功能
- 测试增强后的洗手行为识别
- 输出详细的检测结果和置信度

## 依赖要求

确保安装以下依赖：
```bash
pip install mediapipe opencv-python numpy
```

## 性能优势

1. **准确性提升**：MediaPipe 提供更精确的手部关键点检测
2. **鲁棒性增强**：结合多种检测方法，提高系统稳定性
3. **实时性能**：MediaPipe 优化的推理引擎保证实时处理能力
4. **兼容性好**：保持与现有系统的完全兼容

## 注意事项

- MediaPipe 需要较新的 Python 版本（推荐 3.8+）
- 首次运行时会下载 MediaPipe 模型文件
- 在资源受限的环境中可以选择禁用 MediaPipe 功能

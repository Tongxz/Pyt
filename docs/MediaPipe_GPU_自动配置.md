# MediaPipe GPU 自动配置功能

## 概述

本项目已实现 MediaPipe GPU 智能检测和自动配置功能，系统会根据硬件环境自动选择最佳的运行模式（GPU 或 CPU），无需手动配置。

## 功能特性

### 1. 智能GPU检测
- **自动硬件检测**：系统启动时自动检测 CUDA 可用性
- **显存要求检查**：确保至少有 2GB 可用显存
- **计算能力验证**：要求 GPU 计算能力 ≥ 6.0
- **优雅降级**：不满足条件时自动切换到 CPU 模式

### 2. 配置策略

#### GPU 启用条件
系统会在以下条件**全部满足**时启用 GPU 加速：
1. CUDA 可用（`torch.cuda.is_available() == True`）
2. 可用显存 ≥ 2.0GB
3. GPU 计算能力 ≥ 6.0
4. 未手动禁用 GPU（`MEDIAPIPE_DISABLE_GPU` 环境变量未设置为 `1`、`true` 或 `yes`）

#### CPU 模式触发条件
以下任一条件满足时将使用 CPU 模式：
- CUDA 不可用
- 可用显存不足（< 2.0GB）
- GPU 计算能力不足（< 6.0）
- 手动禁用 GPU
- PyTorch 不可用
- 检测过程中发生异常

### 3. 手动控制

#### 强制禁用 GPU
可以通过设置环境变量强制使用 CPU 模式：
```bash
# Windows
set MEDIAPIPE_DISABLE_GPU=1

# Linux/macOS
export MEDIAPIPE_DISABLE_GPU=1
```

#### 支持的禁用值
- `1`
- `true`
- `yes`

（不区分大小写）

## 技术实现

### 核心函数

```python
def _configure_mediapipe_gpu():
    """智能配置MediaPipe GPU使用策略"""
    try:
        import torch
        
        # 检查CUDA是否可用
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            # 检查GPU显存是否足够（至少需要2GB可用显存）
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory_gb = torch.cuda.memory_allocated(0) / (1024**3)
            available_memory_gb = gpu_memory_gb - allocated_memory_gb
            
            # 检查是否为支持的GPU架构（计算能力>=6.0）
            compute_capability = torch.cuda.get_device_capability(0)
            compute_version = compute_capability[0] + compute_capability[1] * 0.1
            
            # GPU使用条件检查
            manual_disable = os.environ.get('MEDIAPIPE_DISABLE_GPU', '').lower() in ['1', 'true', 'yes']
            
            if not manual_disable and available_memory_gb >= 2.0 and compute_version >= 6.0:
                # 启用GPU加速
                if 'MEDIAPIPE_DISABLE_GPU' in os.environ:
                    del os.environ['MEDIAPIPE_DISABLE_GPU']
                return True
            else:
                # 禁用GPU，使用CPU模式
                os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
                return False
        else:
            # CUDA不可用，使用CPU模式
            os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
            return False
            
    except Exception as e:
        # 异常情况，默认使用CPU模式
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
        return False
```

### 集成模块

#### 1. PoseDetector
- 位置：`src/core/pose_detector.py`
- 功能：姿态检测和手部关键点检测
- GPU配置：自动检测并应用最佳设备配置

#### 2. BehaviorRecognizer
- 位置：`src/core/behavior.py`
- 功能：行为识别和洗手检测
- GPU配置：继承 PoseDetector 的配置策略

## 使用方法

### 1. 自动配置（推荐）

```python
from src.core.pose_detector import PoseDetector
from src.core.behavior import BehaviorRecognizer

# 系统会自动检测并配置最佳设备
pose_detector = PoseDetector(use_mediapipe=True)
behavior_recognizer = BehaviorRecognizer(use_mediapipe=True)
```

### 2. 手动禁用 GPU

```python
import os

# 在导入模块前设置环境变量
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

from src.core.pose_detector import PoseDetector

# 将强制使用CPU模式
pose_detector = PoseDetector(use_mediapipe=True)
```

### 3. 检查当前配置

```python
from src.core.pose_detector import _gpu_enabled

if _gpu_enabled:
    print("MediaPipe 正在使用 GPU 加速")
else:
    print("MediaPipe 正在使用 CPU 模式")
```

## 测试验证

### 运行测试脚本

```bash
# 测试GPU自动配置功能
python test_mediapipe_gpu_config.py

# 检查当前GPU状态
python check_gpu.py
```

### 测试内容

1. **GPU硬件要求检查**
   - CUDA 可用性
   - 显存容量检查
   - 计算能力验证

2. **自动配置测试**
   - PoseDetector 初始化
   - BehaviorRecognizer 初始化
   - 设备模式验证

3. **手动禁用测试**
   - 环境变量设置
   - 强制CPU模式验证

## 性能对比

### GPU 模式优势
- **处理速度**：比 CPU 模式快 3-5 倍
- **并发能力**：支持更高的并发检测任务
- **实时性**：更适合实时视频处理

### CPU 模式特点
- **兼容性**：适用于所有硬件环境
- **稳定性**：避免 GPU 驱动相关问题
- **资源占用**：显存占用为零

## 日志信息

### GPU 启用日志
```
MediaPipe GPU加速已启用 - GPU: NVIDIA GeForce RTX 4090, 可用显存: 23.5GB, 计算能力: 8.9
PoseDetector initialized with MediaPipe (GPU模式)
MediaPipe hands detector initialized successfully with unified params (GPU模式)
```

### CPU 模式日志
```
MediaPipe使用CPU模式 - 原因: 显存不足(1.5GB<2.0GB)
PoseDetector initialized with MediaPipe (CPU模式)
MediaPipe hands detector initialized successfully with unified params (CPU模式)
```

## 故障排除

### 常见问题

#### 1. GPU 未启用
**可能原因：**
- 显存不足
- GPU 计算能力不足
- CUDA 驱动问题

**解决方案：**
```bash
# 检查 CUDA 状态
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 检查显存使用
nvidia-smi

# 运行完整检测
python check_gpu.py
```

#### 2. 意外的 CPU 模式
**检查步骤：**
1. 确认环境变量未设置：`echo $MEDIAPIPE_DISABLE_GPU`
2. 检查硬件要求：运行 `python test_mediapipe_gpu_config.py`
3. 查看详细日志：启用 DEBUG 级别日志

#### 3. 性能问题
**优化建议：**
- 确保 GPU 驱动为最新版本
- 检查显存使用情况，避免显存不足
- 考虑降低模型复杂度（`model_complexity=0`）

## 更新历史

### v1.0.0 (2024-01-21)
- ✅ 实现智能 GPU 检测功能
- ✅ 添加硬件要求验证
- ✅ 支持手动禁用 GPU
- ✅ 集成到 PoseDetector 和 BehaviorRecognizer
- ✅ 添加完整的测试套件
- ✅ 更新部署脚本

## 注意事项

1. **首次运行**：MediaPipe 可能需要下载模型文件，请确保网络连接正常
2. **显存管理**：GPU 模式会占用额外显存，请合理规划显存使用
3. **驱动兼容性**：建议使用最新的 NVIDIA 驱动程序
4. **环境变量**：修改环境变量后需要重启 Python 进程才能生效

## 相关文档

- [MediaPipe 集成说明](MEDIAPIPE_INTEGRATION.md)
- [硬件部署方案](硬件部署方案.md)
- [性能优化指南](性能优化指南.md)
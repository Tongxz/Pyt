# 训练脚本目录

本目录包含模型训练相关的脚本文件。

## 脚本说明

### start_training.sh
- **功能**: Linux/macOS发网检测模型训练启动脚本
- **主要功能**:
  - 自动检查ultralytics依赖
  - 验证数据集完整性
  - 配置训练参数
  - 启动YOLO模型训练
  - 自动创建模型保存目录
- **使用方法**:
  ```bash
  ./start_training.sh [选项]
  ```
- **支持参数**:
  - `--epochs N`: 训练轮数（默认100）
  - `--batch-size N`: 批次大小（默认16）
  - `--img-size N`: 图像尺寸（默认640）
  - `--weights FILE`: 预训练权重（默认models/yolo/yolov8m.pt）
  - `--device STR`: 训练设备（如cuda:0或cpu）

### start_training.ps1
- **功能**: Windows PowerShell发网检测模型训练脚本
- **主要功能**:
  - 检查Python环境（支持conda环境）
  - 验证ultralytics安装
  - 数据集完整性检查
  - 启动模型训练
- **使用方法**:
  ```powershell
  .\start_training.ps1 [参数]
  ```
- **支持参数**:
  - `-Epochs N`: 训练轮数
  - `-BatchSize N`: 批次大小
  - `-ImgSize N`: 图像尺寸
  - `-Weights FILE`: 预训练权重
  - `-Device STR`: 训练设备
  - `-Help`: 显示帮助信息

## 使用前准备

### 1. 数据集准备
确保数据集目录结构正确：
```
datasets/hairnet/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 2. 环境要求
- Python 3.8+
- ultralytics库
- PyTorch（支持CUDA可选）
- 足够的存储空间

### 3. 数据集配置
如果没有准备好的数据集，可以使用：
```bash
python prepare_roboflow_dataset.py --input /path/to/dataset.zip --output datasets/hairnet
```

## 训练示例

### 基础训练
```bash
# Linux/macOS
./training/start_training.sh

# Windows
.\training\start_training.ps1
```

### 自定义参数训练
```bash
# Linux/macOS
./training/start_training.sh --epochs 200 --batch-size 32 --device cuda:0

# Windows
.\training\start_training.ps1 -Epochs 200 -BatchSize 32 -Device "cuda:0"
```

## 训练输出

训练完成后，模型文件将保存在：
- `models/hairnet_model/weights/best.pt` - 最佳模型
- `models/hairnet_model/weights/last.pt` - 最后一轮模型
- `models/hairnet_model/results.csv` - 训练结果

## 相关文档

- [训练指南](../README_TRAINING.md)
- [发网检测文档](../docs/README_HAIRNET_DETECTION.md)
- [数据集添加指南](../docs/README_ADD_DATASET.md)

# 模型文件目录

本目录用于存储项目中使用的各种机器学习模型文件。

## 目录结构

```
models/
├── yolo/                    # YOLO系列目标检测模型
│   ├── models/yolo/yolov8n.pt          # YOLOv8 nano模型
│   ├── models/yolo/yolov8s.pt          # YOLOv8 small模型
│   ├── models/yolo/yolov8m.pt          # YOLOv8 medium模型
│   └── models/yolo/yolov8l.pt          # YOLOv8 large模型
├── hairnet_detection/       # 发网检测相关模型
│   ├── models/hairnet_detection/hairnet_detection.pt # 预训练的发网检测模型
│   └── weights/            # 训练权重文件目录
├── hairnet_model/          # 🔒 用户训练的发网检测模型（受保护）
│   ├── args.yaml           # 训练参数配置
│   ├── results.csv         # 训练结果记录
│   └── weights/            # 用户训练的模型权重
│       ├── best.pt         # 最佳模型权重
│       └── last.pt         # 最新模型权重
├── pose_detection/          # 手部姿态检测模型
├── behavior_recognition/    # 行为识别模型
└── README.md               # 本说明文件
```

## 重要说明

### 🔒 受保护目录

- **`hairnet_model/`**: 此目录包含用户自己训练的发网检测模型，**不会被自动整理脚本移动或修改**
- 该目录中的文件是训练过程的重要成果，请妥善保管
- 如需备份或迁移，请手动操作

### 模型文件管理

1. **自动下载模型**: YOLO系列模型会在首次使用时自动下载
2. **版本控制**: 大型模型文件通过`.gitignore`排除，不会提交到版本控制系统
3. **目录占位**: 使用`.gitkeep`文件确保空目录结构被保留

### 使用方法

#### 在代码中引用模型

```python
# YOLO模型
yolo_model_path = "models/yolo/models/yolo/yolov8n.pt"

# 发网检测模型
hairnet_model_path = "models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt"

# 用户训练的模型
user_trained_model = "models/hairnet_model/weights/best.pt"
```

#### 配置文件中的路径

```yaml
models:
  yolo:
    path: "models/yolo/models/yolo/yolov8n.pt"
  hairnet_detection:
    path: "models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt"
  user_trained:
    path: "models/hairnet_model/weights/best.pt"
```

## 模型整理工具

项目提供了自动模型文件整理脚本：

```bash
python scripts/organize_model_files.py
```

该脚本会：
- 自动发现项目中的模型文件（包括Git忽略的文件）
- 将模型文件移动到规范的目录结构中
- **自动跳过受保护的目录**（如`hairnet_model/`）
- 生成详细的整理报告

## 注意事项

1. **备份重要模型**: 在进行任何批量操作前，请备份重要的训练模型
2. **权限管理**: 确保模型文件具有适当的读写权限
3. **存储空间**: 大型模型文件会占用较多磁盘空间，定期清理不需要的模型
4. **模型更新**: 当模型版本更新时，及时更新相关配置文件

本目录用于存放项目中使用的各种机器学习模型文件。

## 目录结构

```
models/
├── README.md                    # 本说明文件
├── yolo/                        # YOLO相关模型
│   ├── models/yolo/yolov8n.pt              # YOLOv8 nano模型
│   ├── models/yolo/yolov8s.pt              # YOLOv8 small模型
│   └── models/yolo/yolov8m.pt              # YOLOv8 medium模型
├── hairnet_detection/           # 发网检测模型
│   ├── weights/
│   │   └── best.pt             # 发网检测最佳权重
│   └── config.yaml             # 模型配置文件
├── pose_detection/              # 姿态检测模型
│   └── pose_model.pt           # 姿态检测模型
└── behavior_recognition/        # 行为识别模型
    └── behavior_model.pt       # 行为识别模型
```

## 模型文件说明

### YOLO模型
- **models/yolo/yolov8n.pt**: 轻量级人体检测模型，适用于资源受限环境
- **models/yolo/yolov8s.pt**: 小型人体检测模型，平衡速度和精度
- **models/yolo/yolov8m.pt**: 中型人体检测模型，更高精度

### 发网检测模型
- **best.pt**: 训练得到的最佳发网检测权重文件
- **config.yaml**: 模型训练和推理配置

### 姿态检测模型
- **pose_model.pt**: 用于手部姿态检测的模型

### 行为识别模型
- **behavior_model.pt**: 用于洗手行为识别的模型

## 模型下载

### 自动下载
项目提供了自动下载脚本，运行以下命令可自动下载所需模型：

```bash
# 下载YOLO模型
python scripts/setup_dev.sh

# 或者手动下载
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/models/yolo/yolov8n.pt -O models/yolo/models/yolo/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/models/yolo/yolov8s.pt -O models/yolo/models/yolo/yolov8s.pt
```

### 手动下载
如果自动下载失败，可以手动下载模型文件并放置到相应目录。

## 模型使用

### 在代码中引用模型
```python
# 人体检测
model_path = "models/yolo/models/yolo/yolov8s.pt"
detector = HumanDetector(model_path=model_path)

# 发网检测
hairnet_model_path = "models/hairnet_detection/weights/best.pt"
hairnet_detector = HairnetDetector(model_path=hairnet_model_path)
```

### 配置文件中的模型路径
在 `config/unified_params.yaml` 中配置模型路径：
```yaml
human_detection:
  model_path: models/yolo/models/yolo/yolov8s.pt

hairnet_detection:
  model_path: models/hairnet_detection/weights/best.pt
```

## 注意事项

1. **版本控制**: 模型文件通常较大，已在 `.gitignore` 中排除，不会提交到版本控制系统
2. **存储管理**: 建议使用 Git LFS 或其他大文件存储方案管理模型文件
3. **权限设置**: 确保模型文件具有适当的读取权限
4. **备份策略**: 重要的训练模型应该有备份策略
5. **模型更新**: 更新模型时，建议保留旧版本以便回滚

## 模型训练

如需训练自定义模型，请参考：
- [训练文档](../docs/README_TRAINING.md)
- [训练脚本](../scripts/train_hairnet_model.py)

## 性能优化

- 根据硬件配置选择合适大小的模型
- 考虑使用模型量化或剪枝技术减小模型大小
- 对于生产环境，建议使用ONNX格式的优化模型

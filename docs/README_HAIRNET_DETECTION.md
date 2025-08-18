# 发网检测模型训练与使用

本项目提供了使用YOLOv8训练和测试发网检测模型的完整工具集。

## 文件说明

- `prepare_roboflow_dataset.py`: 处理从Roboflow下载的数据集，并将其转换为YOLOv8格式
- `train_hairnet_model.py`: 使用YOLOv8训练发网检测模型
- `test_hairnet_model.py`: 测试训练好的YOLOv8发网检测模型
- `example_usage.py`: 在Python代码中使用训练好的模型的示例
- `start_training.sh`: 训练模型的快速启动脚本
- `start_testing.sh`: 测试模型的快速启动脚本
- `README_TRAINING.md`: 详细的训练指南

## 数据集管理

### 当前数据集状况

根据分析，当前数据集结构如下：
- **训练集 (train)**: 约200+个样本，包含标注文件
- **验证集 (valid)**: 约30+个样本，包含标注文件
- **测试集 (test)**: 约17个样本，包含标注文件
- **类别**: 3个类别 - `hairnet`(发网), `head`(头部), `person`(人员)

### 扩展数据集

#### 使用自动化脚本 (推荐)

```bash
python add_dataset.py
```

脚本提供以下功能：
- **查看统计**: 显示当前数据集的详细统计信息
- **添加数据**: 自动添加新的图片和YOLO格式标注文件
- **格式转换**: 将LabelMe JSON格式转换为YOLO格式
- **验证检查**: 自动验证标注文件格式的正确性

#### 准备新数据

**选项A: 已有YOLO格式标注**
```
新数据目录/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── labels/
    ├── image1.txt
    └── image2.txt
```

**选项B: LabelMe JSON格式**
脚本支持自动转换LabelMe标注格式

## 快速开始

### 1. 准备数据集

从Roboflow下载发网检测数据集，然后使用以下命令处理数据集：

```bash
python prepare_roboflow_dataset.py --input /path/to/roboflow_dataset.zip --output datasets/hairnet
```

### 2. 训练模型

使用以下命令开始训练：

```bash
./start_training.sh
```

或者使用更多参数：

```bash
./start_training.sh --epochs 200 --batch-size 8 --img-size 640 --weights models/yolo/yolov8s.pt --device cuda:0
```

### 3. 测试模型

使用以下命令测试模型：

```bash
./start_testing.sh --source path/to/image.jpg --view-img
```

或者使用更多参数：

```bash
./start_testing.sh --weights models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt --source path/to/video.mp4 --conf-thres 0.3 --view-img
```

### 4. 在代码中使用模型

使用以下命令运行示例代码：

```bash
python example_usage.py --source path/to/image.jpg
```

或者在自己的Python代码中使用：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt')

# 运行推理
results = model('path/to/image.jpg')

# 处理结果
for r in results:
    boxes = r.boxes  # 边界框
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # 边界框坐标
        conf = box.conf[0]  # 置信度
        cls = int(box.cls[0])  # 类别
        print(f"检测到发网: 坐标=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), 置信度={conf:.2f}")
```

## 详细文档

请参阅 [README_TRAINING.md](README_TRAINING.md) 获取更详细的训练指南。

## 依赖项

- Python 3.6+
- ultralytics
- opencv-python
- pyyaml

安装依赖：

```bash
pip install ultralytics opencv-python pyyaml
```

## 目录结构

```
.
├── datasets/
│   └── hairnet/         # 数据集目录
│       ├── train/       # 训练集
│       ├── valid/       # 验证集
│       ├── test/        # 测试集
│       └── data.yaml    # 数据集配置文件
├── models/              # 模型保存目录
├── prepare_roboflow_dataset.py
├── train_hairnet_model.py
├── test_hairnet_model.py
├── example_usage.py
├── start_training.sh
├── start_testing.sh
├── README_HAIRNET_DETECTION.md
└── README_TRAINING.md
```

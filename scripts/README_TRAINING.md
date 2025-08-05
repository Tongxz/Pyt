# 发网检测模型训练指南

本指南将帮助您使用Roboflow数据集训练YOLOv8模型进行发网检测。

## 准备工作

### 1. 安装依赖

确保您已安装所需的依赖包：

```bash
pip install ultralytics pyyaml opencv-python torch torchvision
```

### 2. 准备数据集

从Roboflow下载的数据集通常是ZIP格式，您需要先解压并准备数据集：

```bash
python scripts/prepare_roboflow_dataset.py --input /path/to/your/downloaded/dataset.zip --output ./datasets/hairnet
```

这个脚本会：
- 解压数据集
- 验证数据集结构
- 创建必要的`data.yaml`配置文件

## 训练模型

### 基本训练

使用以下命令开始训练：

```bash
python scripts/train_hairnet_model.py --data ./datasets/hairnet/data.yaml --epochs 100 --batch-size 16 --img-size 640 --model yolov8n.pt
```

参数说明：
- `--data`: 数据集YAML配置文件路径
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--img-size`: 图像尺寸
- `--model`: 基础模型，可选：yolov8n.pt（最小）, yolov8s.pt（小型）, yolov8m.pt（中型）, yolov8l.pt（大型）, yolov8x.pt（超大型）
- `--device`: 训练设备，可选：cpu, cuda, auto（默认）
- `--name`: 实验名称（默认：hairnet_model）
- `--pretrained`: 使用预训练权重（默认开启）
- `--resume`: 从上次检查点恢复训练
- `--save-dir`: 模型保存目录（默认：./models）

### 高级训练选项

#### 使用更大的模型

如果您有足够的计算资源，可以使用更大的YOLOv8模型获得更好的性能：

```bash
python scripts/train_hairnet_model.py --data ./datasets/hairnet/data.yaml --model yolov8m.pt --epochs 150 --batch-size 8
```

#### 从检查点恢复训练

如果训练中断，可以从上次的检查点恢复：

```bash
python scripts/train_hairnet_model.py --data ./datasets/hairnet/data.yaml --resume --name hairnet_model
```

## 模型评估

训练完成后，模型会自动进行验证。您也可以单独运行验证：

```bash
yolo val model=./models/hairnet_model/weights/best.pt data=./datasets/hairnet/data.yaml
```

## 使用训练好的模型

训练完成后，最佳模型将保存在`./models/hairnet_detector.pt`。您可以在项目中使用这个模型：

```python
from src.core.hairnet_detector import HairnetDetector

# 初始化发网检测器，使用训练好的模型
hairnet_detector = HairnetDetector(model_path='./models/hairnet_detector.pt')

# 使用检测器
result = hairnet_detector.detect(image)
```

## 故障排除

### 1. CUDA内存不足

如果遇到CUDA内存不足的错误，尝试减小批次大小或图像尺寸：

```bash
python scripts/train_hairnet_model.py --data ./datasets/hairnet/data.yaml --batch-size 4 --img-size 416
```

### 2. 数据集格式问题

如果数据集格式有问题，请检查数据集结构是否符合YOLOv8要求：

```
dataset/
  ├── train/
  │   ├── images/
  │   │   ├── image1.jpg
  │   │   └── ...
  │   └── labels/
  │       ├── image1.txt
  │       └── ...
  ├── valid/
  │   ├── images/
  │   │   ├── image1.jpg
  │   │   └── ...
  │   └── labels/
  │       ├── image1.txt
  │       └── ...
  └── data.yaml
```

### 3. 训练效果不佳

如果训练效果不佳，可以尝试：

1. 增加训练轮数
2. 使用更大的模型
3. 增加数据增强
4. 调整学习率

## 参考资料

- [YOLOv8官方文档](https://docs.ultralytics.com/)
- [Roboflow Universe](https://universe.roboflow.com/)

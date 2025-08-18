# 发网检测模型训练指南

本指南将帮助您使用Roboflow数据集训练YOLOv8发网检测模型。

## 目录

- [环境准备](#环境准备)
- [数据集准备](#数据集准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [在项目中使用](#在项目中使用)
- [常见问题](#常见问题)

## 环境准备

### 安装依赖

```bash
pip install ultralytics
pip install opencv-python
pip install pyyaml
```

## 数据集准备

### 从Roboflow下载数据集

1. 访问[Roboflow Universe](https://universe.roboflow.com/)搜索发网检测数据集
2. 选择合适的数据集，例如"Hairnet Detection"或"Mask and Hairnet Detection"
3. 下载数据集，选择YOLOv8格式
4. 将下载的ZIP文件保存到本地

### 处理数据集

使用`prepare_roboflow_dataset.py`脚本处理下载的数据集：

```bash
python prepare_roboflow_dataset.py --input /path/to/roboflow_dataset.zip --output datasets/hairnet
```

或者，如果您已经解压了数据集：

```bash
python prepare_roboflow_dataset.py --input /path/to/extracted_roboflow_folder --output datasets/hairnet
```

脚本将：
1. 解压数据集（如果是ZIP文件）
2. 检测数据集格式
3. 组织数据集为YOLOv8格式
4. 创建`data.yaml`配置文件
5. 验证数据集结构和内容

## 模型训练

### 基本训练

使用`train_hairnet_model.py`脚本训练模型：

```bash
python train_hairnet_model.py --data datasets/hairnet/data.yaml --epochs 100 --batch-size 16 --img-size 640
```

### 高级训练选项

您可以调整以下参数以优化训练过程：

- `--epochs`：训练轮数，默认为100
- `--batch-size`：批次大小，默认为16
- `--img-size`：图像大小，默认为640
- `--weights`：初始权重，默认为models/yolo/yolov8n.pt
- `--device`：训练设备，例如cuda:0或cpu
- `--name`：实验名称，默认为hairnet_model
- `--pretrained`：使用预训练权重
- `--resume`：恢复训练
- `--save-dir`：模型保存目录，默认为./models

例如，使用预训练的YOLOv8s模型并在GPU上训练：

```bash
python train_hairnet_model.py --data datasets/hairnet/data.yaml --epochs 200 --batch-size 8 --img-size 640 --weights models/yolo/yolov8s.pt --device cuda:0 --pretrained
```

## 模型测试

使用`test_hairnet_model.py`脚本测试训练好的模型：

```bash
python test_hairnet_model.py --weights models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt --source path/to/test/image.jpg --view-img
```

您可以使用以下输入源：
- 图像文件：`--source path/to/image.jpg`
- 视频文件：`--source path/to/video.mp4`
- 摄像头：`--source 0`

其他可用选项：
- `--conf-thres`：置信度阈值，默认为0.25
- `--iou-thres`：IoU阈值，默认为0.45
- `--device`：推理设备，例如cuda:0或cpu
- `--view-img`：显示结果
- `--save-txt`：保存结果为txt文件
- `--nosave`：不保存图像/视频

## 在项目中使用

训练完成后，模型将保存在`models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt`。您可以在项目中使用此模型：

1. 确保模型文件位于正确的位置：`models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt`
2. 更新配置文件中的模型路径（如果需要）

在Python代码中使用模型：

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

## 常见问题

### 训练时出现CUDA内存不足错误

如果遇到CUDA内存不足错误，请尝试以下解决方案：

1. 减小批次大小：`--batch-size 8`或更小
2. 减小图像大小：`--img-size 416`
3. 使用更小的模型：`--weights models/yolo/yolov8n.pt`

### 模型检测效果不佳

如果模型检测效果不佳，请尝试以下解决方案：

1. 增加训练轮数：`--epochs 200`或更多
2. 使用更大的模型：`--weights models/yolo/yolov8m.pt`或`models/yolo/yolov8l.pt`
3. 增加数据集大小或使用数据增强
4. 调整学习率和优化器参数

### 推理速度慢

如果推理速度慢，请尝试以下解决方案：

1. 使用更小的模型：`models/yolo/yolov8n.pt`
2. 减小图像大小：`--img-size 416`
3. 使用GPU加速：`--device cuda:0`
4. 使用模型量化技术

---

如有任何问题，请参考[Ultralytics YOLOv8文档](https://docs.ultralytics.com/)或提交Issue。

# YOLOv8 发网检测器使用指南

本文档介绍如何使用训练好的 YOLOv8 发网检测模型。系统已完全迁移到 YOLOv8 检测器，不再使用传统检测器。

## 文件说明

- `src/core/yolo_hairnet_detector.py`: YOLOv8 发网检测器实现
- `src/core/hairnet_detection_factory.py`: 发网检测器工厂，用于创建不同类型的发网检测器
- `examples/use_yolo_hairnet_detector.py`: 直接使用 YOLOv8 发网检测器的示例
- `examples/integrate_yolo_detector.py`: 通过工厂模式集成 YOLOv8 发网检测器的示例

## 快速开始

### 1. 确保模型已训练

确保您已经按照 `README_HAIRNET_DETECTION.md` 中的说明训练了 YOLOv8 发网检测模型，并且模型文件位于 `models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt`。

### 2. 直接使用 YOLOv8 发网检测器

```bash
python examples/use_yolo_hairnet_detector.py --image path/to/image.jpg
```

可选参数：
- `--model`: 模型路径，默认为 `models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt`
- `--conf-thres`: 置信度阈值，默认为 0.25
- `--device`: 计算设备，可选 `cpu`, `cuda`, `auto`
- `--save`: 保存结果图像
- `--output`: 输出图像路径，默认为 `result.jpg`

### 3. 通过工厂模式使用 YOLOv8 发网检测器

```bash
python examples/integrate_yolo_detector.py --image path/to/image.jpg
```

可选参数：
- `--detector-type`: 检测器类型，可选 `auto`, `yolo`，默认为 `auto`（两者效果相同）
- `--model`: 模型路径，默认为 `models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt`
- `--conf-thres`: 置信度阈值，默认为 0.25
- `--iou-thres`: IoU阈值，默认为 0.45
- `--device`: 计算设备，可选 `cpu`, `cuda`, `auto`
- `--save`: 保存结果图像
- `--output`: 输出图像路径，默认为 `result.jpg`

## 在代码中使用

### 直接使用 YOLOv8 发网检测器

```python
from src.core.yolo_hairnet_detector import YOLOHairnetDetector

# 初始化检测器
detector = YOLOHairnetDetector(
    model_path='models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt',
    device='auto',
    conf_thres=0.25,
    iou_thres=0.45
)

# 检测图像
image = cv2.imread('path/to/image.jpg')
result = detector.detect(image)

# 处理结果
wearing_hairnet = result['wearing_hairnet']
confidence = result['confidence']
detections = result['detections']
visualization = result['visualization']

# 显示结果
if visualization is not None:
    cv2.imshow('Result', visualization)
    cv2.waitKey(0)
```

### 通过工厂模式使用

```python
from src.core.hairnet_detection_factory import HairnetDetectionFactory

# 创建YOLOv8检测器
detector = HairnetDetectionFactory.create_detector(
    detector_type='yolo',  # 'auto'和'yolo'效果相同
    model_path='models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt',
    device='auto',
    conf_thres=0.25,
    iou_thres=0.45
)

# 检测图像
image = cv2.imread('path/to/image.jpg')

# 使用检测器
result = detector.detect(image)
visualization = result.get('visualization')
wearing_hairnet = result.get('wearing_hairnet', False)
confidence = result.get('confidence', 0.0)
detections = result.get('detections', [])

```

## 在系统中使用

系统已完全迁移到YOLOv8检测器，您可以通过以下方式使用：

1. 直接使用YOLOv8检测器：

```python
from src.core.yolo_hairnet_detector import YOLOHairnetDetector
detector = YOLOHairnetDetector(model_path='models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt', device='auto')
```

2. 使用工厂模式创建检测器：

```python
from src.core.hairnet_detection_factory import HairnetDetectionFactory

# 创建检测器（可以通过环境变量配置参数）
model_path = os.environ.get('HAIRNET_MODEL_PATH', 'models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt')
device = os.environ.get('HAIRNET_DEVICE', 'auto')
conf_thres = float(os.environ.get('HAIRNET_CONF_THRES', '0.25'))
iou_thres = float(os.environ.get('HAIRNET_IOU_THRES', '0.45'))

detector = HairnetDetectionFactory.create_detector(
    detector_type='yolo',
    model_path=model_path,
    device=device,
    conf_thres=conf_thres,
    iou_thres=iou_thres
)

# 使用检测器进行检测
result = detector.detect(image)
```

## YOLOv8检测器的优势

YOLOv8 发网检测器具有以下优势：

1. **直接检测**：无需先检测人体再提取头部区域，一步到位检测发网
2. **高准确率**：YOLOv8 模型在目标检测任务上表现优异，特别是经过针对发网检测的自训练后
3. **快速推理**：YOLOv8 模型经过优化，推理速度快，适合实时应用场景
4. **可视化支持**：提供检测结果的可视化图像，便于调试和验证
5. **灵活配置**：支持多种参数配置，如置信度阈值、IoU阈值、计算设备等

## 故障排除

如果您在使用 YOLOv8 发网检测器时遇到问题，请参考以下解决方案：

### 1. 模型文件不存在

```
FileNotFoundError: [Errno 2] No such file or directory: 'models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt'
```

确保模型文件已下载并放置在正确的路径。

### 2. 检测结果不准确

尝试调整置信度阈值和IoU阈值：

```python
detector = YOLOHairnetDetector(
    model_path='models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt',
    conf_thres=0.4,  # 增加置信度阈值，减少误检
    iou_thres=0.45   # 调整IoU阈值，控制重叠框处理
)
```

### 3. CUDA 不可用或内存不足

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

或

```
RuntimeError: CUDA out of memory
```

将 `device` 参数设置为 `'cpu'`：

```python
detector = YOLOHairnetDetector(model_path='models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt', device='cpu')
```

如果GPU内存不足，也可以尝试使用更小的模型或降低输入图像分辨率。

### 4. 其他常见问题

1. 确保已安装所有依赖项：`pip install -r requirements.txt`
2. 检查模型文件路径是否正确，确保模型文件存在
3. 如果遇到性能问题，可以调整 `conf_thres` 和 `iou_thres` 参数
4. 确保输入图像格式正确，支持常见的图像格式如JPG、PNG等

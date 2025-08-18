# 模型文件整理报告

整理时间: 2025-08-18 16:48:57,294

## 整理统计

- 成功移动: 7 个文件
- 跳过处理: 2 个文件
- 处理错误: 0 个文件

## 成功移动的文件

- /Users/zhou/Code/python/Pyt/models/yolo/yolov8n.pt -> /Users/zhou/Code/python/Pyt/models/yolo/models/yolo/yolov8n.pt
- /Users/zhou/Code/python/Pyt/models/yolo/yolov8s.pt -> /Users/zhou/Code/python/Pyt/models/yolo/models/yolo/yolov8s.pt
- /Users/zhou/Code/python/Pyt/models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt -> /Users/zhou/Code/python/Pyt/models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt
- /Users/zhou/Code/python/Pyt/models/hairnet_model/weights/last.pt -> /Users/zhou/Code/python/Pyt/models/hairnet_detection/weights/last.pt
- /Users/zhou/Code/python/Pyt/models/hairnet_model/weights/best.pt -> /Users/zhou/Code/python/Pyt/models/hairnet_detection/weights/best.pt
- /Users/zhou/Code/python/Pyt/models/yolo/yolov8l.pt -> /Users/zhou/Code/python/Pyt/models/yolo/models/yolo/yolov8l.pt
- /Users/zhou/Code/python/Pyt/models/yolo/yolov8m.pt -> /Users/zhou/Code/python/Pyt/models/yolo/models/yolo/yolov8m.pt

## 跳过的文件

- /Users/zhou/Code/python/Pyt/src/api/models/yolo/yolov8n.pt -> /Users/zhou/Code/python/Pyt/models/yolo/models/yolo/yolov8n.pt (目标已存在)
- /Users/zhou/Code/python/Pyt/src/api/models/yolo/yolov8s.pt -> /Users/zhou/Code/python/Pyt/models/yolo/models/yolo/yolov8s.pt (目标已存在)

## 整理后的目录结构

```
models/
├── yolo/                    # YOLO系列模型
├── hairnet_detection/       # 发网检测模型
│   └── weights/            # 训练权重文件
├── pose_detection/          # 姿态检测模型
├── behavior_recognition/    # 行为识别模型
└── general/                # 其他模型文件
```

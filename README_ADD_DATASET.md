# 增加数据集指南

本指南将帮助您向现有的发网检测模型添加新的训练数据，以提高模型性能。

## 当前数据集状况

根据分析，当前数据集结构如下：
- **训练集 (train)**: 约200+个样本，包含标注文件
- **验证集 (valid)**: 约30+个样本，包含标注文件  
- **测试集 (test)**: 约17个样本，包含标注文件
- **类别**: 3个类别 - `hairnet`(发网), `head`(头部), `person`(人员)

## 方法一：使用自动化脚本 (推荐)

### 1. 运行数据集扩展工具

```bash
python add_dataset.py
```

### 2. 功能说明

脚本提供以下功能：
- **查看统计**: 显示当前数据集的详细统计信息
- **添加数据**: 自动添加新的图片和YOLO格式标注文件
- **格式转换**: 将LabelMe JSON格式转换为YOLO格式
- **验证检查**: 自动验证标注文件格式的正确性

### 3. 准备新数据

#### 选项A: 已有YOLO格式标注
如果您已经有YOLO格式的标注文件：

```
新数据目录/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

#### 选项B: LabelMe JSON格式
如果您使用LabelMe标注工具：

```
新数据目录/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── json_labels/
    ├── image1.json
    ├── image2.json
    └── ...
```

### 4. YOLO标注格式说明

每个`.txt`文件包含图片中所有目标的标注，格式为：
```
class_id center_x center_y width height
```

- `class_id`: 类别ID (0=hairnet, 1=head, 2=person)
- `center_x, center_y`: 边界框中心点坐标 (归一化到0-1)
- `width, height`: 边界框宽度和高度 (归一化到0-1)

示例：
```
0 0.5 0.3 0.2 0.4
1 0.6 0.7 0.15 0.25
```

## 方法二：手动添加数据

### 1. 准备图片文件

将新的图片文件复制到对应目录：
```bash
# 添加到训练集
cp your_images/* datasets/hairnet/train/images/

# 添加到验证集
cp your_images/* datasets/hairnet/valid/images/
```

### 2. 准备标注文件

将对应的标注文件复制到标注目录：
```bash
# 添加到训练集
cp your_labels/* datasets/hairnet/train/labels/

# 添加到验证集
cp your_labels/* datasets/hairnet/valid/labels/
```

### 3. 删除缓存文件

```bash
# 删除训练集缓存
rm -f datasets/hairnet/train/labels/labels.cache

# 删除验证集缓存
rm -f datasets/hairnet/valid/labels/labels.cache
```

## 数据标注建议

### 1. 标注质量要求
- **准确性**: 边界框应紧密包围目标对象
- **一致性**: 相同类型的对象应使用相同的标注标准
- **完整性**: 图片中所有相关目标都应被标注

### 2. 数据多样性
建议添加以下类型的图片：
- **不同角度**: 正面、侧面、背面视角
- **不同光照**: 明亮、昏暗、逆光环境
- **不同场景**: 厨房、工厂、实验室等
- **不同人群**: 不同年龄、性别、体型
- **不同发网**: 不同颜色、材质、样式的发网

### 3. 推荐的数据比例
- **训练集**: 70-80% 的数据
- **验证集**: 15-20% 的数据
- **测试集**: 5-10% 的数据

## 重新训练模型

添加新数据后，建议重新训练模型：

### 1. 从现有模型继续训练
```bash
python train_hairnet_model.py --weights models/hairnet_detection.pt --epochs 50
```

### 2. 调整训练参数
```bash
# 增加训练轮数
python train_hairnet_model.py --epochs 150

# 调整学习率
python train_hairnet_model.py --lr0 0.001

# 增加批次大小（如果GPU内存足够）
python train_hairnet_model.py --batch 32
```

### 3. 使用数据增强
```bash
# 启用更多数据增强
python train_hairnet_model.py --augment
```

## 验证新数据效果

### 1. 检查训练进度
```bash
# 查看训练日志
tail -f models/hairnet_model/results.csv
```

### 2. 测试模型性能
```bash
# 在测试集上评估
python test_with_fixtures.py

# 测试单张图片
python test_example.py
```

### 3. 监控关键指标
- **mAP50**: 在IoU=0.5时的平均精度
- **mAP50-95**: 在IoU=0.5-0.95时的平均精度
- **Precision**: 精确率
- **Recall**: 召回率

## 常见问题解决

### 1. 标注格式错误
```
错误: 坐标不在0-1范围内
解决: 检查标注工具设置，确保输出归一化坐标
```

### 2. 类别ID不匹配
```
错误: 类别ID超出范围
解决: 确保使用正确的类别映射 (0=hairnet, 1=head, 2=person)
```

### 3. 文件名不匹配
```
错误: 图片没有对应的标注文件
解决: 确保图片和标注文件名相同（除了扩展名）
```

### 4. 内存不足
```
错误: CUDA out of memory
解决: 减少batch_size或使用更小的图片尺寸
```

## 最佳实践

1. **逐步添加**: 先添加少量高质量数据，观察效果后再大量添加
2. **平衡数据**: 确保各类别的样本数量相对平衡
3. **质量优于数量**: 高质量的标注比大量低质量数据更有效
4. **定期备份**: 在添加新数据前备份现有模型和数据
5. **监控过拟合**: 注意验证集性能，避免过拟合

## 数据集扩展示例

假设您要添加100张新的发网检测图片：

```bash
# 1. 运行扩展工具
python add_dataset.py

# 2. 选择"添加新的图片和标注"
# 3. 输入图片目录: /path/to/new_images
# 4. 输入标注目录: /path/to/new_labels
# 5. 选择添加到训练集

# 6. 重新训练模型
python train_hairnet_model.py --epochs 120

# 7. 测试新模型
python test_with_fixtures.py
```

通过以上步骤，您可以有效地扩展数据集并提高模型性能。建议在添加大量数据前先进行小规模测试，确保流程正确。
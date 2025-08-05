# 测试数据目录

此目录用于存放测试所需的固定数据（fixtures）。

## 目录结构

```
fixtures/
├── images/       # 测试图像
│   ├── person/   # 人体测试图像
│   └── hairnet/  # 发网测试图像
└── videos/       # 测试视频
```

## 使用说明

1. 将测试图像放在对应的子目录中
2. 在测试代码中使用相对路径引用这些测试数据
3. 使用 pytest fixtures 加载测试数据

## 示例

```python
import pytest
import cv2
from pathlib import Path

@pytest.fixture
def sample_person_image():
    """加载测试人物图像"""
    image_path = Path(__file__).parent.parent / "fixtures" / "images" / "person" / "test_person.jpg"
    return cv2.imread(str(image_path))

@pytest.fixture
def sample_hairnet_image():
    """加载测试发网图像"""
    image_path = Path(__file__).parent.parent / "fixtures" / "images" / "hairnet" / "test_hairnet.jpg"
    return cv2.imread(str(image_path))
```

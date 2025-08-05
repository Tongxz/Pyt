# 测试目录

本目录包含项目的所有测试文件和测试数据。

## 目录结构

```
tests/
├── unit/           # 单元测试
│   ├── test_detector.py
│   ├── test_hairnet_detector.py
│   └── test_math_utils.py
├── integration/    # 集成测试
│   └── test_api_endpoints.py
├── fixtures/       # 测试数据
│   ├── images/     # 测试图像
│   │   ├── person/   # 人体测试图像
│   │   └── hairnet/  # 发网测试图像
│   └── videos/     # 测试视频
└── conftest.py     # pytest 配置和通用 fixtures
```

## 测试分类

### 单元测试 (Unit Tests)

单元测试位于 `unit/` 目录下，用于测试单个组件的功能。每个测试文件对应一个源代码模块。

### 集成测试 (Integration Tests)

集成测试位于 `integration/` 目录下，用于测试多个组件之间的交互，以及API端点的功能。

## 测试数据

测试数据位于 `fixtures/` 目录下，包括：

- `images/person/`: 人体测试图像
- `images/hairnet/`: 发网测试图像
- `videos/`: 测试视频

## 运行测试

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit/

# 运行集成测试
pytest tests/integration/

# 运行特定测试文件
pytest tests/unit/test_detector.py

# 运行特定测试函数
pytest tests/unit/test_detector.py::TestHumanDetector::test_detect_empty_image

# 生成测试覆盖率报告
pytest --cov=src --cov-report=html
```

## 添加新测试

1. 在适当的目录下创建测试文件，命名为 `test_*.py`
2. 使用 pytest 或 unittest 编写测试
3. 使用 `conftest.py` 中定义的 fixtures 加载测试数据

## 测试数据管理

- 测试数据应放在 `fixtures/` 目录下
- 使用 `conftest.py` 中的 fixtures 加载测试数据
- 避免在测试代码中硬编码文件路径

## 项目清理

项目根目录下的测试文件已被移动到适当的测试目录中。如果在开发过程中在根目录创建了临时测试文件，可以使用清理脚本进行整理：

```bash
# 清理项目根目录下的测试文件和测试图像
python scripts/cleanup_tests.py
```

该脚本会：
- 删除根目录下已整理到tests目录的测试文件
- 删除不必要的测试图像文件
- 将脚本文件从根目录移动到scripts目录
- 将数据库文件从根目录移动到data目录

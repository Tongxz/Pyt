# 项目开发规范

## 1. 目录结构规范

所有创建的文件都应该按照项目规范存放在对应的目录下：

### 核心目录
- 所有的 Python 脚本都应该放在 `src/` 目录下
- 所有的测试脚本都应该放在 `tests/` 目录下
- 所有的配置文件都应该放在 `config/` 目录下
- 所有的文档文件都应该放在 `docs/` 目录下

### 资源和数据目录
- 所有的资源文件都应该放在 `resources/` 目录下
- 所有的模型文件都应该放在 `models/` 目录下
- 所有的日志文件都应该放在 `logs/` 目录下
- 所有的临时文件都应该放在 `temp/` 目录下
- 所有的测试数据都存放在 `tests/fixtures/` 目录下

### 环境和部署目录
- 所有的虚拟环境都应该放在 `venv/` 目录下
- 所有的环境变量配置文件都应该放在 `env/` 目录下
- 部署相关文件放在 `deployment/` 目录下
- 开发脚本放在 `scripts/` 目录下

### src/ 目录结构
```
src/
├── api/          # API 相关代码
├── core/         # 核心业务逻辑
├── services/     # 服务层代码
├── utils/        # 工具函数
├── config/       # 配置管理
└── detection/    # 检测相关模块
```

### tests/ 目录结构
```
tests/
├── unit/         # 单元测试
├── integration/  # 集成测试
├── fixtures/     # 测试数据
│   ├── images/   # 测试图像
│   └── videos/   # 测试视频
└── conftest.py   # pytest 配置
```

## 2. 文件命名规范

### Python 文件命名
- 使用小写字母和下划线：`my_module.py`
- 测试文件以 `test_` 开头：`test_my_module.py`
- 类名使用 PascalCase：`class MyClass`
- 函数和变量使用 snake_case：`def my_function()`

### 配置文件命名
- YAML 配置文件：`config_name.yaml`
- JSON 配置文件：`config_name.json`
- 环境变量文件：`.env`, `.env.local`, `.env.production`

### 文档文件命名
- README 文件：`README.md`, `README_FEATURE.md`
- 技术文档使用描述性名称：`技术方案.md`, `部署指南.md`

## 3. 代码风格规范

### Python 代码风格
- 遵循 PEP 8 标准
- 使用 4 个空格缩进，不使用 Tab
- 行长度限制为 88 字符（Black 格式化器标准）
- 使用类型注解：`def function(param: str) -> int:`

### 导入规范
```python
# 标准库导入
import os
import sys
from typing import List, Dict, Optional

# 第三方库导入
import cv2
import numpy as np
import torch

# 本地模块导入
from src.core.detector import HumanDetector
from src.utils.logger import get_logger
```

### 注释和文档字符串
```python
def detect_handwashing(self, person_bbox: List[int],
                      hand_regions: List[Dict],
                      frame: np.ndarray) -> float:
    """
    检测洗手行为

    Args:
        person_bbox: 人体边界框 [x1, y1, x2, y2]
        hand_regions: 手部区域列表
        frame: 输入图像帧

    Returns:
        float: 洗手行为置信度 (0.0-1.0)

    Raises:
        ValueError: 当输入参数无效时
    """
```

## 4. 测试规范

### 测试文件组织
- 单元测试：测试单个函数或类的功能
- 集成测试：测试多个模块间的交互
- 端到端测试：测试完整的业务流程

### 测试数据管理
- **禁止随意创建测试数据**
- 所有测试数据必须存放在 `tests/fixtures/` 目录下
- 测试数据应该版本控制，但大文件使用 Git LFS
- 使用描述性的测试数据文件名

### 测试命名规范
```python
def test_detect_handwashing_with_valid_input():
    """测试有效输入下的洗手检测"""
    pass

def test_detect_handwashing_with_empty_hands():
    """测试无手部输入时的洗手检测"""
    pass
```

## 5. 配置管理规范

### 配置文件层次
1. `config/default.yaml` - 默认配置
2. `config/development.yaml` - 开发环境配置
3. `config/production.yaml` - 生产环境配置
4. `.env` - 环境变量（不提交到版本控制）

### 敏感信息处理
- API 密钥、数据库密码等敏感信息必须使用环境变量
- 不得将敏感信息硬编码在代码中
- 使用 `.env` 文件管理本地开发环境变量

## 6. 日志规范

### 日志级别使用
- `DEBUG`: 详细的调试信息
- `INFO`: 一般信息，程序正常运行
- `WARNING`: 警告信息，程序可以继续运行
- `ERROR`: 错误信息，程序出现问题
- `CRITICAL`: 严重错误，程序可能无法继续运行

### 日志格式
```python
logger.info("检测到 %d 个人体，%d 个手部区域", len(persons), len(hands))
logger.error("模型加载失败: %s", str(e))
```

### 日志文件管理
- 所有日志文件存放在 `logs/` 目录下
- 使用日期轮转：`app_2024-01-01.log`
- 设置合理的日志保留期限

## 7. 安全规范

### 代码安全
- 不得在代码中硬编码密码、API 密钥等敏感信息
- 对用户输入进行验证和清理
- 使用参数化查询防止 SQL 注入
- 定期更新依赖包，修复安全漏洞

### 文件权限
- 配置文件权限设置为 600 或 644
- 可执行脚本权限设置为 755
- 敏感文件不得提交到版本控制系统

## 8. 性能规范

### 代码优化
- 避免在循环中进行重复的昂贵操作
- 合理使用缓存机制
- 对于 I/O 密集型操作使用异步编程
- 及时释放不再使用的资源

### 内存管理
- 大型数据处理时注意内存使用
- 使用生成器处理大量数据
- 及时清理 OpenCV 创建的窗口和资源

## 9. 版本控制规范

### Git 提交规范
```
feat: 添加新功能
fix: 修复 bug
docs: 更新文档
style: 代码格式调整
refactor: 代码重构
test: 添加或修改测试
chore: 构建过程或辅助工具的变动
```

### 分支管理
- `main`: 主分支，稳定版本
- `develop`: 开发分支
- `feature/*`: 功能分支
- `hotfix/*`: 热修复分支

## 10. 部署规范

### 环境隔离
- 开发、测试、生产环境严格隔离
- 使用 Docker 容器化部署
- 环境配置通过环境变量管理

### 监控和日志
- 生产环境必须配置监控和告警
- 日志集中收集和分析
- 定期备份重要数据

## 11. 文档规范

### 必需文档
- `README.md`: 项目概述和快速开始
- `CONTRIBUTING.md`: 贡献指南
- `CHANGELOG.md`: 版本变更记录
- API 文档：详细的接口说明

### 文档更新
- 代码变更时同步更新相关文档
- 使用中文编写用户文档
- 技术文档包含代码示例

## 12. 依赖管理规范

### Python 依赖
- 使用 `requirements.txt` 管理生产依赖
- 使用 `requirements.dev.txt` 管理开发依赖
- 固定版本号避免兼容性问题
- 定期更新依赖包

### 虚拟环境
- 每个项目使用独立的虚拟环境
- 虚拟环境不提交到版本控制
- 提供环境搭建脚本

---

**注意**: 所有开发人员都必须严格遵守以上规范，确保代码质量和项目的可维护性。

# 贡献指南
# Contributing Guide

欢迎为人体行为检测系统项目做出贡献！本文档将指导您如何参与项目开发。

## 📋 目录

- [开发环境搭建](#开发环境搭建)
- [Git 工作流](#git-工作流)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [测试规范](#测试规范)
- [文档规范](#文档规范)
- [发布流程](#发布流程)

## 🛠️ 开发环境搭建

### 方式一：Docker 环境（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd human-behavior-detection

# 启动开发环境
docker-compose --profile development up -d

# 进入开发容器
docker-compose exec dev-tools bash
```

### 方式二：本地环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -e .

# 安装开发工具
pip install pre-commit
pre-commit install
```

### 环境验证

```bash
# 运行测试
pytest

# 检查代码质量
flake8 src/
black --check src/
mypy src/

# 启动应用
python main.py --mode api
```

## 🔄 Git 工作流

我们采用 **Git Flow** 工作流模式：

### 分支策略

- `main`: 生产环境分支，只接受来自 `release` 和 `hotfix` 的合并
- `develop`: 开发主分支，集成所有功能分支
- `feature/*`: 功能开发分支，从 `develop` 创建
- `release/*`: 发布准备分支，从 `develop` 创建
- `hotfix/*`: 紧急修复分支，从 `main` 创建

### 开发流程

1. **创建功能分支**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **开发和提交**
   ```bash
   # 进行开发工作
   git add .
   git commit -m "feat: add new feature"
   ```

3. **推送和创建 PR**
   ```bash
   git push origin feature/your-feature-name
   # 在 GitHub/GitLab 创建 Pull Request
   ```

4. **代码审查和合并**
   - 至少需要 1 人审查
   - 所有 CI 检查必须通过
   - 合并后删除功能分支

### 分支命名规范

- `feature/功能描述`: 新功能开发
- `bugfix/问题描述`: Bug 修复
- `hotfix/紧急修复描述`: 紧急修复
- `refactor/重构描述`: 代码重构
- `docs/文档描述`: 文档更新

## 📁 项目结构与文件组织

### 目录结构

项目采用以下目录结构组织代码和资源：

```
├── src/               # 源代码目录
│   ├── core/          # 核心功能模块
│   ├── config/        # 配置管理
│   ├── utils/         # 工具函数
│   └── api/           # Web API接口
├── tests/             # 测试代码
│   ├── unit/          # 单元测试
│   ├── integration/   # 集成测试
│   └── fixtures/      # 测试数据
├── data/              # 数据目录（存放数据库文件等）
├── models/            # AI模型文件
├── scripts/           # 脚本工具目录
├── docs/              # 文档
├── config/            # 配置文件
└── logs/              # 日志文件
```

### 文件组织规范

- **源代码文件**：所有源代码文件应放在 `src/` 目录下的相应模块中
- **测试文件**：所有测试文件应放在 `tests/` 目录下，按测试类型分类
- **脚本文件**：所有工具脚本应放在 `scripts/` 目录下，不应放在项目根目录
- **数据文件**：所有数据文件（如数据库文件）应放在 `data/` 目录下
- **配置文件**：所有配置文件应放在 `config/` 目录下
- **日志文件**：所有日志文件应输出到 `logs/` 目录下

### 文件命名规范

- **Python文件**：使用小写字母和下划线，如 `file_utils.py`
- **测试文件**：以 `test_` 开头，如 `test_detector.py`
- **配置文件**：使用小写字母和下划线，如 `default_config.yaml`

### 项目清理

定期使用 `scripts/cleanup_tests.py` 脚本清理项目根目录：

```bash
# 清理项目根目录
python scripts/cleanup_tests.py
```

该脚本会：
- 删除根目录下已整理到tests目录的测试文件
- 删除不必要的测试图像文件
- 将脚本文件从根目录移动到scripts目录
- 将数据库文件从根目录移动到data目录

## 📝 代码规范

### Python 代码规范

我们遵循 [PEP 8](https://pep8.org/) 和 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)：

```python
# 好的示例
class PersonDetector:
    """人体检测器类。

    用于检测图像或视频中的人体目标。

    Attributes:
        model_path: 模型文件路径
        confidence_threshold: 置信度阈值
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """初始化检测器。

        Args:
            model_path: 模型文件路径
            confidence_threshold: 置信度阈值，默认0.5
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model = None

    def detect(self, image: np.ndarray) -> List[Detection]:
        """检测图像中的人体。

        Args:
            image: 输入图像，BGR格式

        Returns:
            检测结果列表

        Raises:
            ValueError: 当图像格式不正确时
        """
        if image is None or len(image.shape) != 3:
            raise ValueError("Invalid image format")

        # 检测逻辑
        return self._process_image(image)
```

### 代码质量要求

- **类型注解**: 所有函数参数和返回值必须有类型注解
- **文档字符串**: 所有公共类和函数必须有详细的文档字符串
- **错误处理**: 适当的异常处理和错误信息
- **单一职责**: 每个函数和类只负责一个功能
- **命名规范**: 使用有意义的变量和函数名

### 工具配置

项目使用以下工具确保代码质量：

- **Black**: 代码格式化（行长度88字符）
- **isort**: 导入排序
- **Flake8**: 代码检查
- **MyPy**: 类型检查
- **Bandit**: 安全检查
- **Pre-commit**: 提交前检查

## 💬 提交规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

### 提交消息格式

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### 提交类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式调整（不影响功能）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动
- `ci`: CI/CD 相关

### 示例

```bash
# 新功能
git commit -m "feat(detector): add YOLOv8 person detection"

# Bug 修复
git commit -m "fix(tracker): resolve ID switching issue"

# 文档更新
git commit -m "docs: update API documentation"

# 重大变更
git commit -m "feat!: change detection API interface

BREAKING CHANGE: detection method now returns structured results"
```

## 🧪 测试规范

### 测试结构

```
tests/
├── unit/           # 单元测试
│   ├── test_detector.py
│   ├── test_tracker.py
│   └── test_utils.py
├── integration/    # 集成测试
│   ├── test_api.py
│   └── test_pipeline.py
├── fixtures/       # 测试数据
│   ├── images/
│   └── videos/
└── conftest.py     # pytest 配置
```

### 测试要求

- **覆盖率**: 单元测试覆盖率不低于 80%
- **命名**: 测试函数名要清晰描述测试内容
- **独立性**: 测试之间不能有依赖关系
- **数据**: 使用 fixtures 管理测试数据

### 测试示例

```python
import pytest
from src.core.detector import PersonDetector

class TestPersonDetector:
    """人体检测器测试类。"""

    @pytest.fixture
    def detector(self):
        """创建检测器实例。"""
        return PersonDetector("models/models/yolo/yolov8n.pt")

    @pytest.fixture
    def sample_image(self):
        """加载测试图像。"""
        return cv2.imread("tests/fixtures/images/person.jpg")

    def test_detect_single_person(self, detector, sample_image):
        """测试单人检测。"""
        results = detector.detect(sample_image)

        assert len(results) == 1
        assert results[0].confidence > 0.5
        assert results[0].class_name == "person"

    def test_detect_empty_image(self, detector):
        """测试空图像处理。"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = detector.detect(empty_image)

        assert len(results) == 0

    def test_invalid_image_raises_error(self, detector):
        """测试无效图像抛出异常。"""
        with pytest.raises(ValueError):
            detector.detect(None)
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_detector.py

# 生成覆盖率报告
pytest --cov=src --cov-report=html

# 运行性能测试
pytest -m slow
```

## 📚 文档规范

### 文档类型

- **README**: 项目概述和快速开始
- **API 文档**: 自动生成的 API 参考
- **用户指南**: 详细的使用说明
- **开发文档**: 架构设计和开发指南

### 文档要求

- 使用 Markdown 格式
- 包含代码示例
- 保持更新和准确性
- 支持中英文双语

### 生成文档

```bash
# 生成 API 文档
sphinx-build -b html docs/ docs/_build/

# 启动文档服务器
sphinx-autobuild docs/ docs/_build/
```

## 🚀 发布流程

### 版本号规范

我们使用 [Semantic Versioning](https://semver.org/)：

- `MAJOR.MINOR.PATCH`
- `1.0.0`: 主要版本（不兼容的 API 变更）
- `1.1.0`: 次要版本（向后兼容的功能性新增）
- `1.1.1`: 修订版本（向后兼容的问题修正）

### 发布步骤

1. **创建发布分支**
   ```bash
   git checkout develop
   git checkout -b release/1.1.0
   ```

2. **更新版本信息**
   - 更新 `setup.py` 中的版本号
   - 更新 `CHANGELOG.md`
   - 更新文档

3. **测试和修复**
   ```bash
   pytest
   docker-compose up --build
   ```

4. **合并到主分支**
   ```bash
   git checkout main
   git merge release/1.1.0
   git tag v1.1.0
   git push origin main --tags
   ```

5. **合并回开发分支**
   ```bash
   git checkout develop
   git merge main
   git push origin develop
   ```

## 🤝 代码审查

### 审查清单

- [ ] 代码符合项目规范
- [ ] 有适当的测试覆盖
- [ ] 文档已更新
- [ ] 没有安全问题
- [ ] 性能影响可接受
- [ ] 向后兼容性

### 审查原则

- **建设性**: 提供具体的改进建议
- **及时性**: 在24小时内完成审查
- **学习性**: 分享知识和最佳实践
- **尊重性**: 保持专业和友善的态度

## 📞 获取帮助

如果您在贡献过程中遇到问题，可以通过以下方式获取帮助：

- 创建 Issue 描述问题
- 在 Discussion 中提问
- 联系项目维护者
- 查看项目文档和 Wiki

感谢您的贡献！🎉

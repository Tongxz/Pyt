# 人体行为检测系统

一个基于深度学习的人体行为检测系统，专注于工业环境中的安全合规监控，包括发网佩戴检测、洗手行为识别等功能。

## 功能特性

### 核心功能
- **人体检测**: 基于YOLOv8的实时人体检测
- **发网检测**: 专门的CNN模型检测工作人员是否佩戴发网
- **行为识别**: 洗手、消毒等行为的智能识别
- **区域管理**: 支持多区域监控和行为合规检查
- **实时监控**: WebSocket实时数据推送
- **统计分析**: 详细的检测统计和合规率分析

### 技术特性
- **多模态输入**: 支持图像、视频和实时摄像头
- **高性能**: GPU加速推理，支持批量处理
- **可扩展**: 模块化设计，易于添加新的检测功能
- **数据管理**: SQLite数据库存储检测记录和统计信息
- **RESTful API**: 完整的API接口，支持第三方集成

## 技术栈

- **后端**: FastAPI, Python 3.8+
- **AI模型**: YOLOv8, PyTorch, 自定义CNN
- **数据库**: SQLite
- **前端**: HTML5, CSS3, JavaScript
- **部署**: Docker, Uvicorn, Gunicorn
- **测试**: pytest, unittest

## 系统架构

```
├── src/
│   ├── api/                 # FastAPI应用
│   │   └── app.py          # 主应用文件
│   ├── core/               # 核心检测模块
│   │   ├── detector.py     # 人体检测器
│   │   ├── hairnet_detector.py  # 发网检测器
│   │   ├── yolo_hairnet_detector.py # YOLO发网检测器
│   │   └── data_manager.py # 数据管理
│   ├── config/             # 配置模块
│   └── utils/              # 工具函数
├── docs/                   # 技术文档
│   ├── README_HAIRNET_DETECTION.md  # 发网检测文档
│   ├── README_ADD_DATASET.md        # 数据集添加指南
│   ├── README_YOLO_INTEGRATION.md   # YOLO集成文档
│   ├── README_WEB_TESTING.md        # Web测试文档
│   ├── 技术方案.md                   # 技术方案文档
│   ├── 项目执行方案.md               # 项目执行方案
│   └── 敏捷迭代执行方案.md           # 敏捷迭代方案
├── deployment/             # 部署脚本
│   └── deploy_win.bat     # Windows部署脚本
├── development/            # 开发环境脚本
│   ├── setup_dev_env.sh   # 环境配置脚本
│   └── start_dev.sh       # 开发启动脚本
├── training/               # 训练脚本
│   ├── start_training.sh   # Linux/macOS训练脚本
│   └── start_training.ps1  # Windows训练脚本
├── testing/                # 测试脚本
│   ├── start_testing.sh    # 模型测试脚本
│   └── test_api_curl.sh    # API测试脚本
│   📝 **注意**: 所有脚本已修复路径引用问题，可从任意位置执行
├── frontend/               # 前端界面
├── models/                 # 模型文件和训练结果
├── datasets/               # 训练数据集
├── tests/                  # 测试代码
│   ├── unit/              # 单元测试
│   ├── integration/       # 集成测试
│   └── fixtures/          # 测试数据
└── scripts/                # 开发工具脚本
```

## 🏗️ 系统架构

```
├── src/
│   ├── core/           # 核心检测模块
│   │   ├── detector.py     # 人体检测器
│   │   ├── tracker.py      # 多目标追踪
│   │   ├── behavior.py     # 行为识别
│   │   ├── data_manager.py # 数据管理器
│   │   └── region.py       # 区域管理
│   ├── config/         # 配置管理
│   ├── utils/          # 工具函数
│   └── api/            # Web API接口
├── models/             # AI模型文件
├── data/               # 数据目录（存放数据库文件等）
├── logs/               # 日志文件
├── scripts/            # 脚本工具目录
└── config/             # 配置文件
```

## 🛠️ 开发环境搭建

### 依赖安装顺序（macOS/CPU 或 Apple Silicon）
1. 先安装 *PyTorch*（官方夜间 CPU 版即可满足 `ultralytics` 依赖）：
   ```bash
   pip install --pre torch torchvision torchaudio \
       --extra-index-url https://download.pytorch.org/whl/nightly/cpu
   ```
   - 如需 GPU/CUDA，请前往 [PyTorch 官网安装向导](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本。
2. **再安装其他依赖**（包含 `ultralytics`）：
   ```bash
   pip install -r requirements.dev.txt
   ```
   > 提示：若 `pip` 解析器仍报告冲突，可先执行 `pip install ultralytics --no-deps`，再单独安装 `opencv-python` 等依赖。

### 常见安装问题与解决
| 症状 | 根因 | 解决方案 |
|------|------|----------|
| `ResolutionImpossible` 与 `torch` 冲突 | 先安装的 *ultralytics* 触发了对旧版 `torch>=1.7.0` 的解析 | **先装 torch**，或使用 `--no-deps` 安装 *ultralytics* |
| 找不到 `torch` 版本 | macOS 需使用 *nightly CPU* 索引 | 添加 `--extra-index-url https://download.pytorch.org/whl/nightly/cpu` |
| MPS/GPU 不可用 | Apple Silicon 默认 CPU 版 | 升级到 macOS ≥ 12.3 并使用 `--pre` 安装，或改用 CUDA 版 |

> 完整依赖见 `requirements.dev.txt`，生产镜像仍使用根目录 `requirements.txt`。

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 4GB+ RAM
- 摄像头或视频文件

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd human-behavior-detection

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 自训练模型

#### 模型训练

```bash
# 训练发网检测模型
python train_hairnet_model.py

# 使用自定义配置训练
python train_hairnet_model.py --epochs 100 --batch-size 16
```

#### 数据集管理

```bash
# 准备Roboflow数据集
python prepare_roboflow_dataset.py

# 添加新的训练数据
python add_dataset.py --images path/to/images --labels path/to/labels
```

#### 模型测试

```bash
# 测试训练好的模型
python test_hairnet_model.py

# 测试指定模型文件
python test_hairnet_model.py --model models/hairnet_model/weights/best.pt
```

更多详细信息请参考 [docs/README_HAIRNET_DETECTION.md](docs/README_HAIRNET_DETECTION.md)

### 基本使用

#### 1. 实时检测模式

```bash
# 使用默认摄像头
python main.py --mode detection --source 0

# 使用视频文件
python main.py --mode detection --source path/to/video.mp4

# 启用调试模式
python main.py --mode detection --source 0 --debug
```

#### 2. API服务模式

```bash
# 启动API服务器
python main.py --mode api --port 5000

# 自定义主机和端口
python main.py --mode api --host 0.0.0.0 --port 8080
```

#### 3. 演示模式

```bash
# 运行演示
python main.py --mode demo
```

## 📋 配置说明

### 系统配置

主要配置文件位于 `config/` 目录：

- `default.yaml`: 默认系统配置
- `models.yaml`: AI模型配置
- `cameras.yaml`: 摄像头配置

### 检测配置

```yaml
detection:
  confidence_threshold: 0.5
  iou_threshold: 0.4
  max_detections: 100

tracking:
  max_disappeared: 30
  max_distance: 50

behavior:
  enabled_behaviors:
    - hairnet_detection
    - handwash_detection
    - sanitize_detection
```

## 🔧 API接口

### 健康检查

```bash
GET /health
```

### 检测接口

```bash
# 上传图片检测
POST /api/v1/detection/image
Content-Type: multipart/form-data

# 实时视频流检测
WS /api/v1/detection/stream
```

### 配置管理

```bash
# 获取配置
GET /api/v1/config

# 更新配置
PUT /api/v1/config
Content-Type: application/json
```

## 🎮 开发指南

### 项目结构

```
src/
├── core/               # 核心功能模块
│   ├── __init__.py
│   ├── detector.py     # 人体检测
│   ├── tracker.py      # 目标追踪
│   ├── behavior.py     # 行为识别
│   └── region.py       # 区域管理
├── config/             # 配置管理
│   ├── __init__.py
│   ├── settings.py     # 系统设置
│   ├── model_config.py # 模型配置
│   └── camera_config.py# 摄像头配置
├── utils/              # 工具函数
│   ├── __init__.py
│   ├── logger.py       # 日志工具
│   ├── image_utils.py  # 图像处理
│   ├── video_utils.py  # 视频处理
│   ├── math_utils.py   # 数学工具
│   └── file_utils.py   # 文件工具
└── api/                # Web API
    ├── __init__.py
    ├── app.py          # Flask应用
    └── routes/         # 路由定义
```

### 添加新的行为检测

1. 在 `src/core/behavior.py` 中添加新的行为类型
2. 实现对应的检测逻辑
3. 更新配置文件
4. 添加相应的测试

### 自定义检测区域

1. 使用 `RegionManager` 类管理检测区域
2. 配置区域类型和规则
3. 设置行为合规性检查

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_detector.py

# 生成覆盖率报告
pytest --cov=src tests/
```

## 🛠️ 脚本工具

项目中的脚本工具位于 `scripts/` 目录下，用于辅助开发、测试和维护工作。

### 主要脚本工具

#### 项目清理工具

```bash
# 清理项目根目录，移动脚本文件到scripts目录，移动数据库文件到data目录
python scripts/cleanup_tests.py
```

`cleanup_tests.py` 脚本用于：
- 删除已整理到tests目录的根目录测试文件
- 删除不必要的测试图像文件
- 将脚本文件从根目录移动到scripts目录
- 将数据库文件从根目录移动到data目录

#### 其他工具脚本

- `analyze_detection_parameters.py`: 分析检测参数
- `debug_detection_parameters.py`: 调试检测参数
- `enhanced_roi_visualizer.py`: 增强ROI可视化
- `improved_head_roi.py`: 改进头部ROI提取
- `view_enhanced_results.py`: 查看增强结果
- `view_improved_roi.py`: 查看改进的ROI
- `view_roi_results.py`: 查看ROI结果
- `visualize_roi.py`: ROI可视化

## 📊 性能优化

### GPU加速

确保安装了CUDA版本的PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 模型优化

- 使用TensorRT进行模型加速
- 调整输入分辨率平衡精度和速度
- 启用多线程处理

## 🐛 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头权限
   - 确认摄像头索引正确
   - 尝试不同的摄像头索引

2. **模型加载失败**
   - 检查模型文件路径
   - 确认模型文件完整性
   - 检查CUDA环境配置

3. **检测精度低**
   - 调整置信度阈值
   - 检查光照条件
   - 考虑重新训练模型

## 📝 更新日志

### v1.0.0 (2024-01-XX)

- ✨ 初始版本发布
- 🎯 基础人体检测功能
- 🔄 多目标追踪系统
- 🎭 行为识别模块
- 🌐 Web API接口
- 📊 实时监控界面

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目链接: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 邮箱: [your-email@example.com]

## 🙏 致谢

- [YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测框架
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [Flask](https://flask.palletsprojects.com/) - Web框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

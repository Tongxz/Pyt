# 人体行为检测系统

基于深度学习的实时人体行为检测与分析系统，专注于食品安全场景下的行为监控。

## 🎯 项目特性

- **实时检测**: 基于YOLOv8的高性能人体检测
- **多目标追踪**: 支持多人同时追踪和行为分析
- **行为识别**: 发网佩戴、洗手、手部消毒等行为检测
- **区域管理**: 灵活的检测区域配置和规则引擎
- **自学习能力**: 持续优化的AI模型
- **Web API**: RESTful API接口，易于集成
- **实时监控**: 支持多路视频流处理

## 🏗️ 系统架构

```
├── src/
│   ├── core/           # 核心检测模块
│   │   ├── detector.py     # 人体检测器
│   │   ├── tracker.py      # 多目标追踪
│   │   ├── behavior.py     # 行为识别
│   │   └── region.py       # 区域管理
│   ├── config/         # 配置管理
│   ├── utils/          # 工具函数
│   └── api/            # Web API接口
├── models/             # AI模型文件
├── data/               # 数据目录
├── logs/               # 日志文件
└── config/             # 配置文件
```

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
pytest tests/test_detector.py

# 生成覆盖率报告
pytest --cov=src tests/
```

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
# 开发环境脚本使用指南

本项目提供了两个开发环境脚本，用于自动化开发环境的设置、检测和管理。

## 脚本概览

### 1. `check_dev_env.py` - Python环境检测脚本

功能强大的Python环境检测工具，提供详细的环境分析报告。

**主要功能：**
- ✅ Python版本检查
- ✅ 虚拟环境状态检测
- ✅ 依赖包版本检查
- ✅ GPU支持检测
- ✅ 模型文件检查
- ✅ 模块导入测试
- ✅ 生成详细的JSON报告

**使用方法：**
```bash
# 运行环境检测
python check_dev_env.py

# 或者直接执行
./check_dev_env.py
```

**输出示例：**
```
开发环境检测脚本
项目: 人体行为检测系统

============================================================
                        Python版本检查
============================================================

[INFO] 当前Python版本: 3.10.13
[INFO] 要求Python版本: >= 3.10.13
[SUCCESS] Python版本符合要求

============================================================
                         依赖包检查
============================================================

[INFO] 检查关键依赖包:
------------------------------------------------------------
✓ torch               2.2.2           (>= any)
✓ ultralytics         8.1.34          (>= any)
✗ torchaudio          未安装           (需要 any)
------------------------------------------------------------
```

### 2. `start_dev.sh` - 开发环境启动脚本

一站式开发环境管理工具，集成环境检测、依赖安装和服务启动。

**主要功能：**
- 🔧 自动激活虚拟环境
- 📋 Python版本验证
- 📦 自动安装缺失依赖
- 🚀 启动后端和前端服务
- 📊 服务状态监控
- 🛑 服务停止和重启

**使用方法：**
```bash
# 启动开发环境（默认）
./start_dev.sh
./start_dev.sh start

# 仅检测环境
./start_dev.sh check

# 仅安装依赖
./start_dev.sh install

# 查看服务状态
./start_dev.sh status

# 停止所有服务
./start_dev.sh stop

# 重启所有服务
./start_dev.sh restart

# 显示帮助
./start_dev.sh help
```

## 详细功能说明

### 环境检测功能

#### Python版本检查
- 检查当前Python版本是否符合项目要求
- 从`.python-version`或`pyproject.toml`读取版本要求
- 显示Python可执行文件路径和系统路径

#### 虚拟环境检测
- 自动检测是否在虚拟环境中运行
- 查找项目目录下的虚拟环境（venv、.venv、env、.env）
- 提供虚拟环境激活命令

#### 依赖包检查
- 解析`requirements.txt`和`requirements.dev.txt`
- 检查关键依赖包的安装状态和版本
- 识别缺失和过时的包
- 提供安装和更新命令

#### GPU支持检测
- 检查PyTorch的CUDA支持
- 显示GPU设备信息
- 提供CPU/GPU运行建议

#### 模型文件检查
- 检查训练好的模型文件是否存在
- 显示模型文件大小
- 识别缺失的模型文件

#### 模块导入测试
- 测试关键模块的导入状态
- 计算导入成功率
- 识别导入错误

### 服务管理功能

#### 后端服务
- 自动设置环境变量（模型路径、设备配置等）
- 检测端口占用情况
- 后台启动FastAPI服务
- 提供API文档访问地址

#### 前端服务
- 启动静态文件服务器
- 自动处理端口冲突
- 提供前端访问地址

#### 服务监控
- 实时检查服务运行状态
- 记录进程ID到文件
- 提供日志文件路径
- 支持优雅停止和强制终止

## 配置文件说明

### 项目配置文件

1. **`.python-version`** - Python版本要求
   ```
   3.10.13
   ```

2. **`requirements.txt`** - 生产环境依赖
   ```
   torch>=2.0.0
   ultralytics>=8.0.0
   fastapi>=0.100.0
   ...
   ```

3. **`requirements.dev.txt`** - 开发环境依赖
   ```
   black
   pytest
   flake8
   ...
   ```

4. **`pyproject.toml`** - 项目元数据和构建配置
   ```toml
   [project]
   name = "human-behavior-detection"
   requires-python = ">=3.8"
   dependencies = [...]
   ```

### 生成的文件

1. **`dev_env_report.json`** - 环境检测报告
   ```json
   {
     "timestamp": "2024-01-01 12:00:00",
     "python_version": {
       "current": "3.10.13",
       "required": "3.10.13",
       "ok": true
     },
     "dependencies": {
       "satisfied": [...],
       "missing": [...],
       "outdated": [...]
     },
     "overall_status": true
   }
   ```

2. **`backend.pid`** - 后端服务进程ID
3. **`frontend.pid`** - 前端服务进程ID
4. **`backend.log`** - 后端服务日志
5. **`frontend.log`** - 前端服务日志

## 环境变量配置

脚本会自动设置以下环境变量：

```bash
export HAIRNET_MODEL_PATH="models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt"
export HAIRNET_DEVICE="cpu"
export HAIRNET_CONF_THRES="0.5"
export HAIRNET_IOU_THRES="0.4"
```

## 故障排除

### 常见问题

1. **Python版本不符合要求**
   ```bash
   # 使用pyenv安装指定版本
   pyenv install 3.10.13
   pyenv local 3.10.13
   ```

2. **虚拟环境创建失败**
   ```bash
   # 手动创建虚拟环境
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **依赖安装失败**
   ```bash
   # 升级pip
   python -m pip install --upgrade pip

   # 分别安装依赖
   pip install torch torchvision
   pip install ultralytics
   pip install -r requirements.txt
   ```

4. **端口被占用**
   ```bash
   # 查看端口占用
   lsof -i :8000
   lsof -i :8080

   # 终止进程
   kill -9 <PID>
   ```

5. **模型文件缺失**
   ```bash
   # 下载YOLOv8模型
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/models/yolo/yolov8n.pt

   # 或运行训练脚本
   python scripts/train_hairnet_model.py
   ```

### 日志查看

```bash
# 查看后端日志
tail -f backend.log

# 查看前端日志
tail -f frontend.log

# 查看环境报告
cat dev_env_report.json | python -m json.tool
```

## 最佳实践

1. **首次使用**
   ```bash
   # 1. 检测环境
   ./start_dev.sh check

   # 2. 安装依赖
   ./start_dev.sh install

   # 3. 启动服务
   ./start_dev.sh start
   ```

2. **日常开发**
   ```bash
   # 启动开发环境
   ./start_dev.sh

   # 开发完成后停止服务
   ./start_dev.sh stop
   ```

3. **问题调试**
   ```bash
   # 检查环境状态
   ./start_dev.sh status

   # 重新检测环境
   python check_dev_env.py

   # 重启服务
   ./start_dev.sh restart
   ```

## 脚本特性

### 安全特性
- 🔒 检查命令执行权限
- 🛡️ 优雅处理进程终止
- 📝 详细的错误日志记录
- ⚠️ 用户确认重要操作

### 用户体验
- 🎨 彩色输出和清晰的状态指示
- 📊 进度显示和详细反馈
- 🔄 自动重试和错误恢复
- 📖 完整的帮助文档

### 兼容性
- 🍎 支持macOS
- 🐧 支持Linux
- 🐍 支持Python 3.8+
- 📦 支持多种包管理器

## 贡献指南

如果您想改进这些脚本，请：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

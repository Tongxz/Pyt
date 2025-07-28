#!/bin/bash
# 开发环境快速设置脚本
# Development Environment Quick Setup Script

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."
    
    # 检查Python版本
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 未安装"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d" " -f2 | cut -d"." -f1,2)
    required_version="3.8"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        log_error "Python 版本需要 >= 3.8，当前版本: $python_version"
        exit 1
    fi
    
    log_success "Python 版本检查通过: $python_version"
    
    # 检查Docker
    if command -v docker &> /dev/null; then
        log_success "Docker 已安装"
    else
        log_warning "Docker 未安装，将跳过Docker相关设置"
    fi
    
    # 检查Git
    if command -v git &> /dev/null; then
        log_success "Git 已安装"
    else
        log_error "Git 未安装"
        exit 1
    fi
}

# 创建虚拟环境
setup_venv() {
    log_info "设置Python虚拟环境..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "虚拟环境创建成功"
    else
        log_info "虚拟环境已存在"
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    log_success "pip 已升级到最新版本"
}

# 安装依赖
install_dependencies() {
    log_info "安装项目依赖..."
    
    # 安装生产依赖
    pip install -r requirements.txt
    log_success "生产依赖安装完成"
    
    # 安装开发依赖
    pip install pytest pytest-cov pytest-mock pytest-asyncio
    pip install black flake8 mypy isort bandit
    pip install pre-commit
    pip install sphinx sphinx-rtd-theme
    log_success "开发依赖安装完成"
    
    # 安装项目包
    pip install -e .
    log_success "项目包安装完成"
}

# 设置Git hooks
setup_git_hooks() {
    log_info "设置Git hooks..."
    
    if [ -f ".pre-commit-config.yaml" ]; then
        pre-commit install
        log_success "Pre-commit hooks 安装完成"
    else
        log_warning "未找到 .pre-commit-config.yaml 文件"
    fi
}

# 创建必要的目录
create_directories() {
    log_info "创建项目目录结构..."
    
    directories=(
        "data/images"
        "data/videos"
        "data/models"
        "data/temp"
        "logs"
        "models"
        "config/local"
        "tests/fixtures/images"
        "tests/fixtures/videos"
        "docs/_build"
        "notebooks"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    log_success "目录结构创建完成"
}

# 设置配置文件
setup_config() {
    log_info "设置配置文件..."
    
    # 创建本地配置文件
    if [ ! -f "config/local/local.yaml" ]; then
        cat > config/local/local.yaml << EOF
# 本地开发配置
# Local Development Configuration

system:
  debug: true
  log_level: "DEBUG"
  
api:
  host: "0.0.0.0"
  port: 8000
  reload: true
  
database:
  url: "sqlite:///data/local.db"
  
redis:
  url: "redis://localhost:6379/0"
  
models:
  yolo_model_path: "models/yolov8n.pt"
  
cameras:
  default_camera:
    source: 0  # 使用默认摄像头
    resolution: [640, 480]
    fps: 30
EOF
        log_success "本地配置文件创建完成"
    else
        log_info "本地配置文件已存在"
    fi
    
    # 创建环境变量文件
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# 环境变量配置
# Environment Variables Configuration

ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# 数据库配置
DATABASE_URL=sqlite:///data/local.db

# Redis配置
REDIS_URL=redis://localhost:6379/0

# API配置
API_HOST=0.0.0.0
API_PORT=8000

# 模型配置
MODEL_PATH=models/yolov8n.pt

# 安全配置
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
EOF
        log_success "环境变量文件创建完成"
    else
        log_info "环境变量文件已存在"
    fi
}

# 下载模型文件
download_models() {
    log_info "下载预训练模型..."
    
    if [ ! -f "models/yolov8n.pt" ]; then
        log_info "下载 YOLOv8 nano 模型..."
        python -c "
import urllib.request
import os
os.makedirs('models', exist_ok=True)
urllib.request.urlretrieve(
    'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'models/yolov8n.pt'
)
print('YOLOv8 模型下载完成')
"
        log_success "YOLOv8 模型下载完成"
    else
        log_info "YOLOv8 模型已存在"
    fi
}

# 运行测试
run_tests() {
    log_info "运行初始测试..."
    
    if pytest tests/unit/ -v --tb=short; then
        log_success "单元测试通过"
    else
        log_warning "部分测试失败，请检查代码"
    fi
}

# 设置Docker环境
setup_docker() {
    if command -v docker &> /dev/null; then
        log_info "设置Docker环境..."
        
        # 构建开发镜像
        if docker build -f Dockerfile.dev -t hbd:dev . > /dev/null 2>&1; then
            log_success "Docker开发镜像构建完成"
        else
            log_warning "Docker镜像构建失败"
        fi
        
        # 创建Docker网络
        if ! docker network ls | grep -q hbd-network; then
            docker network create hbd-network > /dev/null 2>&1
            log_success "Docker网络创建完成"
        fi
    fi
}

# 生成开发文档
generate_docs() {
    log_info "生成开发文档..."
    
    if [ -d "docs" ]; then
        cd docs
        if command -v sphinx-build &> /dev/null; then
            sphinx-build -b html . _build > /dev/null 2>&1
            log_success "文档生成完成"
        else
            log_warning "Sphinx未安装，跳过文档生成"
        fi
        cd ..
    fi
}

# 显示使用说明
show_usage() {
    log_success "开发环境设置完成！"
    echo
    echo "使用说明:"
    echo "1. 激活虚拟环境: source venv/bin/activate"
    echo "2. 启动API服务: python main.py --mode api"
    echo "3. 启动演示模式: python main.py --mode demo"
    echo "4. 运行测试: pytest"
    echo "5. 代码格式化: black src/ tests/"
    echo "6. 代码检查: flake8 src/ tests/"
    echo "7. 启动Docker环境: docker-compose --profile development up -d"
    echo "8. 查看更多命令: make help"
    echo
    echo "配置文件位置:"
    echo "- 主配置: config/default.yaml"
    echo "- 本地配置: config/local/local.yaml"
    echo "- 环境变量: .env"
    echo
    echo "文档地址:"
    echo "- 项目文档: docs/_build/index.html"
    echo "- API文档: http://localhost:8000/docs (启动服务后)"
    echo
    log_info "开始愉快的开发吧！🚀"
}

# 主函数
main() {
    echo "======================================"
    echo "  人体行为检测系统开发环境设置"
    echo "  Human Behavior Detection System"
    echo "  Development Environment Setup"
    echo "======================================"
    echo
    
    check_requirements
    setup_venv
    install_dependencies
    setup_git_hooks
    create_directories
    setup_config
    download_models
    run_tests
    setup_docker
    generate_docs
    show_usage
}

# 错误处理
trap 'log_error "设置过程中发生错误，请检查上面的错误信息"' ERR

# 运行主函数
main "$@"
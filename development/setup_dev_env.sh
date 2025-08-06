#!/bin/bash

# 开发环境测试启动脚本
# 自动激活虚拟环境、验证版本、检测依赖并安装缺失依赖

set -e  # 遇到错误时退出

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

# 项目根目录 (脚本现在在development子目录中，需要回到上级目录)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

log_info "开始设置开发环境..."
log_info "项目根目录: $PROJECT_ROOT"

# 检查Python版本
check_python_version() {
    log_info "检查Python版本..."

    if [ -f ".python-version" ]; then
        REQUIRED_VERSION=$(cat .python-version)
        log_info "项目要求Python版本: $REQUIRED_VERSION"
    else
        REQUIRED_VERSION="3.10"
        log_warning "未找到.python-version文件，使用默认版本: $REQUIRED_VERSION"
    fi

    if command -v python3 &> /dev/null; then
        CURRENT_VERSION=$(python3 --version | cut -d' ' -f2)
        log_info "当前Python版本: $CURRENT_VERSION"

        # 简单版本比较（主要检查主版本号和次版本号）
        REQUIRED_MAJOR=$(echo $REQUIRED_VERSION | cut -d'.' -f1)
        REQUIRED_MINOR=$(echo $REQUIRED_VERSION | cut -d'.' -f2)
        CURRENT_MAJOR=$(echo $CURRENT_VERSION | cut -d'.' -f1)
        CURRENT_MINOR=$(echo $CURRENT_VERSION | cut -d'.' -f2)

        if [ "$CURRENT_MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$CURRENT_MINOR" -ge "$REQUIRED_MINOR" ]; then
            log_success "Python版本符合要求"
        else
            log_error "Python版本不符合要求，需要 >= $REQUIRED_VERSION，当前版本: $CURRENT_VERSION"
            exit 1
        fi
    else
        log_error "未找到python3命令"
        exit 1
    fi
}

# 检查并创建虚拟环境
setup_virtual_env() {
    log_info "检查虚拟环境..."

    VENV_DIR="venv"

    if [ ! -d "$VENV_DIR" ]; then
        log_info "创建虚拟环境..."
        python3 -m venv "$VENV_DIR"
        log_success "虚拟环境创建成功"
    else
        log_info "虚拟环境已存在"
    fi

    # 激活虚拟环境
    log_info "激活虚拟环境..."
    source "$VENV_DIR/bin/activate"
    log_success "虚拟环境已激活"

    # 升级pip
    log_info "升级pip..."
    pip install --upgrade pip
}

# 检查依赖文件
check_requirements_files() {
    log_info "检查依赖文件..."

    if [ -f "requirements.txt" ]; then
        log_success "找到requirements.txt"
        REQUIREMENTS_FILE="requirements.txt"
    elif [ -f "requirements.dev.txt" ]; then
        log_success "找到requirements.dev.txt"
        REQUIREMENTS_FILE="requirements.dev.txt"
    elif [ -f "pyproject.toml" ]; then
        log_success "找到pyproject.toml"
        REQUIREMENTS_FILE="pyproject.toml"
    else
        log_error "未找到依赖文件 (requirements.txt, requirements.dev.txt, 或 pyproject.toml)"
        exit 1
    fi
}

# 检查已安装的包
check_installed_packages() {
    log_info "检查已安装的包..."

    # 获取已安装包列表
    pip list --format=freeze > installed_packages.tmp

    log_info "当前已安装的主要包:"
    echo "----------------------------------------"

    # 检查关键包
    KEY_PACKAGES=("torch" "torchvision" "ultralytics" "opencv-python" "fastapi" "numpy" "pillow")

    for package in "${KEY_PACKAGES[@]}"; do
        if pip show "$package" &> /dev/null; then
            VERSION=$(pip show "$package" | grep Version | cut -d' ' -f2)
            log_success "$package: $VERSION"
        else
            log_warning "$package: 未安装"
        fi
    done

    echo "----------------------------------------"
    rm -f installed_packages.tmp
}

# 安装依赖
install_dependencies() {
    log_info "安装项目依赖..."

    if [ "$REQUIREMENTS_FILE" = "pyproject.toml" ]; then
        log_info "使用pip安装pyproject.toml中的依赖..."
        pip install -e .
    else
        log_info "从$REQUIREMENTS_FILE安装依赖..."

        # 过滤掉注释行和空行
        grep -v '^#' "$REQUIREMENTS_FILE" | grep -v '^$' > filtered_requirements.tmp

        # 逐行安装，跳过有问题的包
        while IFS= read -r line; do
            if [[ $line == *"#"* ]]; then
                # 跳过注释掉的行
                log_warning "跳过注释的依赖: $line"
                continue
            fi

            package_name=$(echo "$line" | cut -d'>' -f1 | cut -d'=' -f1 | cut -d'<' -f1)

            log_info "安装: $line"
            if pip install "$line"; then
                log_success "成功安装: $package_name"
            else
                log_warning "安装失败，跳过: $package_name"
            fi
        done < filtered_requirements.tmp

        rm -f filtered_requirements.tmp
    fi
}

# 验证关键依赖
verify_key_dependencies() {
    log_info "验证关键依赖..."

    # 测试导入关键模块
    python3 -c "
import sys
try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
except ImportError:
    print('✗ PyTorch: 未安装或导入失败')

try:
    import cv2
    print(f'✓ OpenCV: {cv2.__version__}')
except ImportError:
    print('✗ OpenCV: 未安装或导入失败')

try:
    import ultralytics
    print(f'✓ Ultralytics: {ultralytics.__version__}')
except ImportError:
    print('✗ Ultralytics: 未安装或导入失败')

try:
    import fastapi
    print(f'✓ FastAPI: {fastapi.__version__}')
except ImportError:
    print('✗ FastAPI: 未安装或导入失败')

try:
    import numpy as np
    print(f'✓ NumPy: {np.__version__}')
except ImportError:
    print('✗ NumPy: 未安装或导入失败')
"
}

# 创建启动脚本
create_startup_scripts() {
    log_info "创建启动脚本..."

    # 创建开发服务器启动脚本
    cat > start_dev_server.sh << 'EOF'
#!/bin/bash
# 开发服务器启动脚本

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "虚拟环境已激活"
else
    echo "警告: 未找到虚拟环境，请先运行 ./setup_dev_env.sh"
fi

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export HAIRNET_MODEL_PATH="models/hairnet_detection.pt"

echo "启动后端API服务器..."
python src/api/app.py
EOF

    chmod +x start_dev_server.sh
    log_success "创建了 start_dev_server.sh"

    # 创建前端服务器启动脚本
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
# 前端服务器启动脚本

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "启动前端服务器..."
cd frontend && python -m http.server 8080
EOF

    chmod +x start_frontend.sh
    log_success "创建了 start_frontend.sh"
}

# 显示使用说明
show_usage() {
    echo ""
    log_success "开发环境设置完成！"
    echo ""
    echo "使用说明:"
    echo "----------------------------------------"
    echo "1. 激活虚拟环境:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. 启动后端服务器:"
    echo "   ./start_dev_server.sh"
    echo "   或者: HAIRNET_MODEL_PATH=models/hairnet_detection.pt python src/api/app.py"
    echo ""
    echo "3. 启动前端服务器:"
    echo "   ./start_frontend.sh"
    echo "   或者: cd frontend && python -m http.server 8080"
    echo ""
    echo "4. 访问应用:"
    echo "   前端: http://localhost:8080"
    echo "   后端API: http://localhost:8000"
    echo "   API文档: http://localhost:8000/docs"
    echo "----------------------------------------"
}

# 主函数
main() {
    log_info "=== 开发环境自动设置脚本 ==="

    check_python_version
    setup_virtual_env
    check_requirements_files
    check_installed_packages
    install_dependencies
    verify_key_dependencies
    create_startup_scripts
    show_usage

    log_success "所有步骤完成！"
}

# 运行主函数
main "$@"

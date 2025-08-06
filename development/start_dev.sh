#!/bin/bash

# 开发环境启动脚本
# 自动检测环境、安装依赖、启动服务

# 设置项目根目录 (脚本现在在development子目录中，需要回到上级目录)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

log_header() {
    echo -e "\n${PURPLE}============================================================${NC}"
    echo -e "${PURPLE}$(printf "%*s" $(((60+${#1})/2)) "$1")${NC}"
    echo -e "${PURPLE}============================================================${NC}\n"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查Python版本
check_python_version() {
    log_header "Python版本检查"

    if ! command_exists python3; then
        log_error "Python3 未安装"
        return 1
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    REQUIRED_VERSION="3.8.0"

    if [ -f ".python-version" ]; then
        REQUIRED_VERSION=$(cat .python-version)
    fi

    log_info "当前Python版本: $PYTHON_VERSION"
    log_info "要求Python版本: >= $REQUIRED_VERSION"

    # 简单版本比较
    if python3 -c "import sys; exit(0 if sys.version_info >= tuple(map(int, '$REQUIRED_VERSION'.split('.'))) else 1)"; then
        log_success "Python版本符合要求"
        return 0
    else
        log_error "Python版本不符合要求"
        return 1
    fi
}

# 检查并激活虚拟环境
setup_virtual_environment() {
    log_header "虚拟环境设置"

    # 检查是否已在虚拟环境中
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        log_success "已在虚拟环境中: $VIRTUAL_ENV"
        return 0
    fi

    # 查找虚拟环境目录
    VENV_DIRS=("venv" ".venv" "env" ".env")
    VENV_PATH=""

    for dir in "${VENV_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            VENV_PATH="$dir"
            break
        fi
    done

    if [ -n "$VENV_PATH" ]; then
        log_info "发现虚拟环境: $VENV_PATH"
        log_info "激活虚拟环境..."

        # 激活虚拟环境
        source "$VENV_PATH/bin/activate"

        if [[ "$VIRTUAL_ENV" != "" ]]; then
            log_success "虚拟环境激活成功"
            return 0
        else
            log_error "虚拟环境激活失败"
            return 1
        fi
    else
        log_warning "未找到虚拟环境"
        read -p "是否创建新的虚拟环境? (y/n): " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "创建虚拟环境..."
            python3 -m venv venv

            if [ $? -eq 0 ]; then
                log_success "虚拟环境创建成功"
                source venv/bin/activate
                log_success "虚拟环境激活成功"
                return 0
            else
                log_error "虚拟环境创建失败"
                return 1
            fi
        else
            log_warning "跳过虚拟环境创建"
            return 1
        fi
    fi
}

# 运行环境检测
run_environment_check() {
    log_header "运行环境检测"

    if [ -f "check_dev_env.py" ]; then
        log_info "运行环境检测脚本..."
        python check_dev_env.py
        ENV_CHECK_RESULT=$?

        if [ $ENV_CHECK_RESULT -eq 0 ]; then
            log_success "环境检测通过"
            return 0
        else
            log_warning "环境检测发现问题"
            return 1
        fi
    else
        log_warning "环境检测脚本不存在，跳过检测"
        return 0
    fi
}

# 安装依赖
install_dependencies() {
    log_header "依赖安装"

    # 检查requirements文件
    REQ_FILES=("requirements.dev.txt" "requirements.txt")
    REQ_FILE=""

    for file in "${REQ_FILES[@]}"; do
        if [ -f "$file" ]; then
            REQ_FILE="$file"
            break
        fi
    done

    if [ -z "$REQ_FILE" ]; then
        log_warning "未找到requirements文件"
        return 1
    fi

    log_info "使用依赖文件: $REQ_FILE"

    # 升级pip
    log_info "升级pip..."
    python -m pip install --upgrade pip

    # 安装依赖
    log_info "安装依赖包..."
    pip install -r "$REQ_FILE"

    if [ $? -eq 0 ]; then
        log_success "依赖安装完成"
        return 0
    else
        log_error "依赖安装失败"
        return 1
    fi
}

# 检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
        return 0  # 端口被占用
    else
        return 1  # 端口空闲
    fi
}

# 启动后端服务
start_backend() {
    log_header "启动后端服务"

    BACKEND_PORT=8000

    # 检查端口
    if check_port $BACKEND_PORT; then
        log_warning "端口 $BACKEND_PORT 已被占用"
        read -p "是否终止占用进程并重启? (y/n): " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "终止占用端口 $BACKEND_PORT 的进程..."
            lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null
            sleep 2
        else
            log_info "跳过后端启动"
            return 0
        fi
    fi

    # 设置环境变量
    export HAIRNET_MODEL_PATH="models/hairnet_detection.pt"
    export HAIRNET_DEVICE="cpu"
    export HAIRNET_CONF_THRES="0.5"
    export HAIRNET_IOU_THRES="0.4"

    log_info "启动后端服务..."
    log_info "模型路径: $HAIRNET_MODEL_PATH"
    log_info "设备: $HAIRNET_DEVICE"

    # 在后台启动后端
    nohup python src/api/app.py > backend.log 2>&1 &
    BACKEND_PID=$!

    # 等待服务启动
    sleep 3

    # 检查服务是否启动成功
    if kill -0 $BACKEND_PID 2>/dev/null; then
        log_success "后端服务启动成功 (PID: $BACKEND_PID)"
        log_info "后端地址: http://localhost:$BACKEND_PORT"
        log_info "API文档: http://localhost:$BACKEND_PORT/docs"
        echo $BACKEND_PID > backend.pid
        return 0
    else
        log_error "后端服务启动失败"
        return 1
    fi
}

# 启动前端服务
start_frontend() {
    log_header "启动前端服务"

    FRONTEND_PORT=8080

    if [ ! -d "frontend" ]; then
        log_warning "前端目录不存在，跳过前端启动"
        return 0
    fi

    # 检查端口
    if check_port $FRONTEND_PORT; then
        log_warning "端口 $FRONTEND_PORT 已被占用"
        read -p "是否终止占用进程并重启? (y/n): " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "终止占用端口 $FRONTEND_PORT 的进程..."
            lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null
            sleep 2
        else
            log_info "跳过前端启动"
            return 0
        fi
    fi

    log_info "启动前端服务..."

    # 在后台启动前端
    cd frontend
    nohup python -m http.server $FRONTEND_PORT > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..

    # 等待服务启动
    sleep 2

    # 检查服务是否启动成功
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        log_success "前端服务启动成功 (PID: $FRONTEND_PID)"
        log_info "前端地址: http://localhost:$FRONTEND_PORT"
        echo $FRONTEND_PID > frontend.pid
        return 0
    else
        log_error "前端服务启动失败"
        return 1
    fi
}

# 显示服务状态
show_status() {
    log_header "服务状态"

    echo "运行中的服务:"
    echo "----------------------------------------"

    # 检查后端
    if [ -f "backend.pid" ]; then
        BACKEND_PID=$(cat backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            echo -e "${GREEN}✓${NC} 后端服务 (PID: $BACKEND_PID) - http://localhost:8000"
        else
            echo -e "${RED}✗${NC} 后端服务 (已停止)"
            rm -f backend.pid
        fi
    else
        echo -e "${RED}✗${NC} 后端服务 (未启动)"
    fi

    # 检查前端
    if [ -f "frontend.pid" ]; then
        FRONTEND_PID=$(cat frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            echo -e "${GREEN}✓${NC} 前端服务 (PID: $FRONTEND_PID) - http://localhost:8080"
        else
            echo -e "${RED}✗${NC} 前端服务 (已停止)"
            rm -f frontend.pid
        fi
    else
        echo -e "${RED}✗${NC} 前端服务 (未启动)"
    fi

    echo "----------------------------------------"
    echo
    echo "日志文件:"
    echo "  - 后端日志: backend.log"
    echo "  - 前端日志: frontend.log"
    echo "  - 环境报告: dev_env_report.json"
    echo
    echo "停止服务: ./start_dev.sh stop"
    echo "查看状态: ./start_dev.sh status"
}

# 停止服务
stop_services() {
    log_header "停止服务"

    # 停止后端
    if [ -f "backend.pid" ]; then
        BACKEND_PID=$(cat backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            log_info "停止后端服务 (PID: $BACKEND_PID)..."
            kill $BACKEND_PID
            sleep 2
            if kill -0 $BACKEND_PID 2>/dev/null; then
                log_warning "强制停止后端服务..."
                kill -9 $BACKEND_PID
            fi
            log_success "后端服务已停止"
        fi
        rm -f backend.pid
    fi

    # 停止前端
    if [ -f "frontend.pid" ]; then
        FRONTEND_PID=$(cat frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            log_info "停止前端服务 (PID: $FRONTEND_PID)..."
            kill $FRONTEND_PID
            sleep 2
            if kill -0 $FRONTEND_PID 2>/dev/null; then
                log_warning "强制停止前端服务..."
                kill -9 $FRONTEND_PID
            fi
            log_success "前端服务已停止"
        fi
        rm -f frontend.pid
    fi
}

# 显示帮助信息
show_help() {
    echo -e "${CYAN}开发环境启动脚本${NC}"
    echo -e "${CYAN}项目: 人体行为检测系统${NC}"
    echo
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  start     启动开发环境 (默认)"
    echo "  stop      停止所有服务"
    echo "  restart   重启所有服务"
    echo "  status    查看服务状态"
    echo "  check     仅运行环境检测"
    echo "  install   仅安装依赖"
    echo "  help      显示帮助信息"
    echo
    echo "示例:"
    echo "  $0              # 启动开发环境"
    echo "  $0 start        # 启动开发环境"
    echo "  $0 stop         # 停止所有服务"
    echo "  $0 status       # 查看服务状态"
    echo "  $0 check        # 仅检测环境"
}

# 主函数
main() {
    local action=${1:-start}

    case $action in
        "start")
            log_header "开发环境启动"

            # 检查Python版本
            if ! check_python_version; then
                log_error "Python版本检查失败，退出"
                exit 1
            fi

            # 设置虚拟环境
            if ! setup_virtual_environment; then
                log_error "虚拟环境设置失败，退出"
                exit 1
            fi

            # 运行环境检测
            run_environment_check
            ENV_CHECK_RESULT=$?

            # 如果环境检测失败，询问是否安装依赖
            if [ $ENV_CHECK_RESULT -ne 0 ]; then
                read -p "环境检测发现问题，是否安装缺失的依赖? (y/n): " -n 1 -r
                echo

                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    if ! install_dependencies; then
                        log_error "依赖安装失败，退出"
                        exit 1
                    fi

                    # 重新检测环境
                    log_info "重新检测环境..."
                    run_environment_check
                fi
            fi

            # 启动服务
            start_backend
            start_frontend

            # 显示状态
            show_status
            ;;

        "stop")
            stop_services
            ;;

        "restart")
            stop_services
            sleep 2
            $0 start
            ;;

        "status")
            show_status
            ;;

        "check")
            check_python_version
            setup_virtual_environment
            run_environment_check
            ;;

        "install")
            check_python_version
            setup_virtual_environment
            install_dependencies
            ;;

        "help")
            show_help
            ;;

        *)
            log_error "未知选项: $action"
            show_help
            exit 1
            ;;
    esac
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

@echo off
REM 一键部署脚本，自动适配 conda 或 venv 环境，适用于 Windows 11 + 4090 GPU
REM 请以管理员身份运行（右键以管理员身份运行命令提示符）
REM 更新日期: 2024-01-20
REM 更新内容: 修复脚本路径、更新模型配置、添加MediaPipe环境变量设置

REM 设置控制台编码为UTF-8以正确显示中文
chcp 65001 >nul

setlocal enabledelayedexpansion

REM 设置项目根目录 (脚本现在在deployment子目录中，需要回到上级目录)
cd /d "%~dp0.."

echo ========================================
echo 人体行为检测系统 - Windows 部署脚本
echo ========================================
echo.

REM 检查 conda 是否可用（支持多种安装路径）
set "CONDA_FOUND=0"
set "CONDA_PATH="

REM 检查常见的conda安装路径
if exist "C:\ProgramData\anaconda3\Scripts\conda.exe" (
    set "CONDA_PATH=C:\ProgramData\anaconda3\Scripts\conda.exe"
    set "CONDA_FOUND=1"
) else if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
    set "CONDA_PATH=%USERPROFILE%\anaconda3\Scripts\conda.exe"
    set "CONDA_FOUND=1"
) else if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
    set "CONDA_PATH=%USERPROFILE%\miniconda3\Scripts\conda.exe"
    set "CONDA_FOUND=1"
) else (
    where conda >nul 2>nul
    if !errorlevel!==0 (
        set "CONDA_PATH=conda"
        set "CONDA_FOUND=1"
    )
)

if !CONDA_FOUND!==1 (
    echo [INFO] 检测到 conda，正在创建/激活 conda 环境...
    "!CONDA_PATH!" env list | findstr /C:"pyt-env" >nul
    if !errorlevel!==0 (
        echo [INFO] 已存在 pyt-env 环境，直接激活
    ) else (
        echo [INFO] 创建新的 conda 环境 pyt-env（Python 3.10.13）
        "!CONDA_PATH!" create -y -n pyt-env python=3.10.13
        if !errorlevel! neq 0 (
            echo [ERROR] conda 环境创建失败！
            pause
            exit /b 1
        )
    )
    echo [INFO] 激活 pyt-env 环境...
    call "!CONDA_PATH!" activate pyt-env
    if !errorlevel! neq 0 (
        echo [WARNING] conda activate 失败，尝试使用完整路径...
        set "PYTHON_PATH=C:\Users\%USERNAME%\.conda\envs\pyt-env\python.exe"
        if exist "!PYTHON_PATH!" (
            echo [INFO] 使用环境路径: !PYTHON_PATH!
        ) else (
            echo [ERROR] 无法找到 pyt-env 环境的 Python 解释器！
            pause
            exit /b 1
        )
    )
) else (
    echo [INFO] 未检测到 conda，使用 venv 创建虚拟环境...
    if not exist .venv (
        python -m venv .venv
        if !errorlevel! neq 0 (
            echo [ERROR] venv 创建失败！请检查 Python 安装。
            pause
            exit /b 1
        )
    )
    call .venv\Scripts\activate
    if !errorlevel! neq 0 (
        echo [ERROR] venv 激活失败！
        pause
        exit /b 1
    )
)

echo.
echo [INFO] 开始安装依赖包...
echo ========================================

REM 检查是否需要使用特定的Python路径
if defined PYTHON_PATH (
    set "PIP_CMD="!PYTHON_PATH!" -m pip"
    set "PYTHON_CMD="!PYTHON_PATH!""
) else (
    set "PIP_CMD=pip"
    set "PYTHON_CMD=python"
)

REM 升级pip
echo [INFO] 升级 pip...
!PIP_CMD! install --upgrade pip
if !errorlevel! neq 0 (
    echo [WARNING] pip 升级失败，继续安装...
)

REM 安装 CUDA 相关依赖（需提前安装好 NVIDIA 驱动和 CUDA Toolkit）
echo [INFO] 正在安装 PyTorch（自动适配 CUDA）...
!PIP_CMD! install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if !errorlevel! neq 0 (
    echo [WARNING] CUDA版本PyTorch安装失败，尝试安装CPU版本...
    !PIP_CMD! install torch torchvision torchaudio
    if !errorlevel! neq 0 (
        echo [ERROR] PyTorch 安装失败！
        pause
        exit /b 1
    )
)

REM 安装项目依赖
echo [INFO] 正在安装项目依赖...
if exist requirements.dev.txt (
    echo [INFO] 使用 requirements.dev.txt 安装开发依赖...
    !PIP_CMD! install -r requirements.dev.txt
) else if exist requirements.txt (
    echo [INFO] 使用 requirements.txt 安装依赖...
    !PIP_CMD! install -r requirements.txt
) else (
    echo [WARNING] 未找到依赖文件，手动安装核心包...
    !PIP_CMD! install ultralytics fastapi uvicorn opencv-python numpy pillow mediapipe
)

if !errorlevel! neq 0 (
    echo [ERROR] 依赖安装失败！
    pause
    exit /b 1
)

REM 验证ultralytics安装
echo [INFO] 验证 ultralytics 安装...
!PYTHON_CMD! scripts\check_ultralytics.py
if !errorlevel! neq 0 (
    echo [ERROR] ultralytics 验证失败！
    pause
    exit /b 1
)

REM 创建models目录结构
echo [INFO] 创建模型目录结构...
if not exist models\yolo (
    mkdir models\yolo
    echo [INFO] 已创建 models\yolo 目录
)

REM 检查 models/yolo/yolov8s.pt 是否存在
echo [INFO] 检查 YOLO 模型文件...
if not exist models\yolo\yolov8s.pt (
    echo [INFO] 正在下载 yolov8s.pt 权重文件...
    !PYTHON_CMD! -c "from ultralytics import YOLO; model = YOLO('yolov8s.pt'); import shutil; shutil.move('yolov8s.pt', 'models/yolo/yolov8s.pt')"
    if !errorlevel! neq 0 (
        echo [WARNING] YOLO模型下载失败，服务启动时会自动下载
    )
) else (
    echo [INFO] YOLO模型文件已存在
)

echo.
echo ========================================
echo [INFO] 启动 API 服务...
echo ========================================

REM 检查API文件是否存在
if not exist src\api\app.py (
    echo [ERROR] 未找到 API 文件 src\api\app.py！
    echo [INFO] 请确保项目结构完整
    pause
    exit /b 1
)

REM 设置MediaPipe环境变量（系统将自动检测GPU可用性）
echo [INFO] 设置MediaPipe环境变量...
REM MEDIAPIPE_DISABLE_GPU 已移除，系统将自动检测GPU并选择最佳设备
set GLOG_logtostderr=1
set GLOG_v=1

REM 启动服务
echo [INFO] 正在启动服务，请稍候...
echo [INFO] 服务地址: http://localhost:8000
echo [INFO] API文档: http://localhost:8000/docs
echo [INFO] 按 Ctrl+C 停止服务
echo.

if defined PYTHON_PATH (
    "!PYTHON_PATH!" -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
) else (
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
)

if !errorlevel! neq 0 (
    echo [ERROR] 服务启动失败！
    echo [INFO] 请检查端口8000是否被占用
    pause
    exit /b 1
)

echo.
echo [INFO] 部署完成！请用浏览器访问 http://localhost:8000/docs 进行接口测试。
endlocal
pause

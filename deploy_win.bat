@echo off
REM 一键部署脚本，自动适配 conda 或 venv 环境，适用于 Windows 11 + 4090 GPU
REM 请以管理员身份运行（右键以管理员身份运行命令提示符）

setlocal enabledelayedexpansion

REM 检查 conda 是否可用
where conda >nul 2>nul
if %errorlevel%==0 (
    echo 检测到 conda，正在创建/激活 conda 环境...
    conda env list | findstr /C:"hairnet_env" >nul
    if %errorlevel%==0 (
        echo 已存在 hairnet_env 环境，直接激活
    ) else (
        echo 创建新的 conda 环境 hairnet_env
        conda create -y -n hairnet_env python=3.10
    )
    call conda activate hairnet_env
) else (
    echo 未检测到 conda，使用 venv 创建虚拟环境...
    if not exist .venv (
        python -m venv .venv
    )
    call .venv\Scripts\activate
)

REM 安装 CUDA 相关依赖（需提前安装好 NVIDIA 驱动和 CUDA Toolkit）
echo 正在安装 PyTorch（自动适配 CUDA）...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM 安装项目依赖
echo 正在安装项目依赖...
pip install -r requirements.txt
pip install ultralytics

REM 检查 yolov8m.pt 是否存在
if not exist yolov8m.pt (
    echo 正在下载 yolov8m.pt 权重...
    python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
)

REM 启动服务
echo 启动 API 服务...
start /b uvicorn src.api.app:app --host 0.0.0.0 --port 8000

echo 部署完成！请用浏览器访问 http://localhost:8000/docs 进行接口测试。
endlocal

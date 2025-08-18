#!/bin/bash

# 发网检测模型训练启动脚本

# 设置项目根目录 (脚本现在在training子目录中，需要回到上级目录)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 检查ultralytics是否已安装
if ! python -c "import ultralytics" &> /dev/null; then
    echo "正在安装ultralytics..."
    pip install ultralytics
fi

# 检查数据集目录
if [ ! -d "datasets/hairnet/train/images" ] || [ ! -d "datasets/hairnet/valid/images" ]; then
    echo "错误: 数据集目录不完整，请先准备数据集"
    echo "使用以下命令准备数据集:"
    echo "python prepare_roboflow_dataset.py --input /path/to/roboflow_dataset.zip --output datasets/hairnet"
    exit 1
fi

# 检查数据集配置文件
if [ ! -f "datasets/hairnet/data.yaml" ]; then
    echo "错误: 数据集配置文件不存在，请先准备数据集"
    exit 1
fi

# 创建模型保存目录
mkdir -p models

# 设置训练参数
EPOCHS=100
BATCH_SIZE=16
IMG_SIZE=640
WEIGHTS="models/yolo/yolov8m.pt"
DEVICE=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --epochs)
            EPOCHS="$2"
            shift
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --img-size)
            IMG_SIZE="$2"
            shift
            shift
            ;;
        --weights)
            WEIGHTS="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        --help)
            echo "用法: ./start_training.sh [选项]"
            echo "选项:"
            echo "  --epochs N       训练轮数，默认为100"
            echo "  --batch-size N   批次大小，默认为16"
            echo "  --img-size N     图像大小，默认为640"
            echo "  --weights FILE   初始权重，默认为models/yolo/yolov8n.pt"
            echo "  --device STR     训练设备，例如cuda:0或cpu"
            echo "  --help           显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $key"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 构建训练命令
COMMAND="python train_hairnet_model.py --data datasets/hairnet/data.yaml --epochs $EPOCHS --batch-size $BATCH_SIZE --img-size $IMG_SIZE --weights $WEIGHTS"

# 添加设备参数（如果指定）
if [ -n "$DEVICE" ]; then
    COMMAND="$COMMAND --device $DEVICE"
fi

# 显示训练信息
echo "=== 发网检测模型训练 ==="
echo "数据集: datasets/hairnet"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "图像大小: $IMG_SIZE"
echo "初始权重: $WEIGHTS"
if [ -n "$DEVICE" ]; then
    echo "训练设备: $DEVICE"
fi
echo

# 确认开始训练
read -p "是否开始训练? [y/N] " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "已取消训练"
    exit 0
fi

# 开始训练
echo "开始训练..."
echo "执行命令: $COMMAND"
echo
$COMMAND

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "训练完成!"
    echo "模型保存在: models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt"
    echo "使用以下命令测试模型:"
    echo "python test_hairnet_model.py --weights models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt --source path/to/test/image.jpg --view-img"
else
    echo "训练失败，请检查错误信息"
fi

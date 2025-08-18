#!/bin/bash

# 发网检测模型测试启动脚本

# 设置项目根目录 (脚本现在在testing子目录中，需要回到上级目录)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 检查ultralytics是否已安装
if ! python -c "import ultralytics" &> /dev/null; then
    echo "正在安装ultralytics..."
    pip install ultralytics
fi

# 设置默认参数
WEIGHTS="models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt"
SOURCE=""
CONF_THRES=0.25
IOU_THRES=0.45
DEVICE=""
VIEW_IMG=false
SAVE_TXT=false
NOSAVE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --weights)
            WEIGHTS="$2"
            shift
            shift
            ;;
        --source)
            SOURCE="$2"
            shift
            shift
            ;;
        --conf-thres)
            CONF_THRES="$2"
            shift
            shift
            ;;
        --iou-thres)
            IOU_THRES="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        --view-img)
            VIEW_IMG=true
            shift
            ;;
        --save-txt)
            SAVE_TXT=true
            shift
            ;;
        --nosave)
            NOSAVE=true
            shift
            ;;
        --help)
            echo "用法: ./start_testing.sh [选项]"
            echo "选项:"
            echo "  --weights FILE     模型权重路径，默认为models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt"
            echo "  --source PATH      输入源，可以是图像、视频路径或摄像头编号(0)"
            echo "  --conf-thres N     置信度阈值，默认为0.25"
            echo "  --iou-thres N      IoU阈值，默认为0.45"
            echo "  --device STR       推理设备，例如cuda:0或cpu"
            echo "  --view-img         显示结果"
            echo "  --save-txt         保存结果为txt文件"
            echo "  --nosave           不保存图像/视频"
            echo "  --help             显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $key"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查模型文件
if [ ! -f "$WEIGHTS" ]; then
    echo "错误: 模型文件不存在: $WEIGHTS"
    echo "请先训练模型或指定正确的模型路径"
    echo "使用以下命令训练模型:"
    echo "./start_training.sh"
    exit 1
fi

# 检查输入源
if [ -z "$SOURCE" ]; then
    echo "错误: 未指定输入源"
    echo "请使用 --source 参数指定输入源，例如:"
    echo "./start_testing.sh --source path/to/image.jpg"
    echo "./start_testing.sh --source path/to/video.mp4"
    echo "./start_testing.sh --source 0  # 使用摄像头"
    exit 1
fi

# 构建测试命令
COMMAND="python test_hairnet_model.py --weights $WEIGHTS --source $SOURCE --conf-thres $CONF_THRES --iou-thres $IOU_THRES"

# 添加可选参数
if [ -n "$DEVICE" ]; then
    COMMAND="$COMMAND --device $DEVICE"
fi

if [ "$VIEW_IMG" = true ]; then
    COMMAND="$COMMAND --view-img"
fi

if [ "$SAVE_TXT" = true ]; then
    COMMAND="$COMMAND --save-txt"
fi

if [ "$NOSAVE" = true ]; then
    COMMAND="$COMMAND --nosave"
fi

# 显示测试信息
echo "=== 发网检测模型测试 ==="
echo "模型: $WEIGHTS"
echo "输入源: $SOURCE"
echo "置信度阈值: $CONF_THRES"
echo "IoU阈值: $IOU_THRES"
if [ -n "$DEVICE" ]; then
    echo "推理设备: $DEVICE"
fi
if [ "$VIEW_IMG" = true ]; then
    echo "显示结果: 是"
fi
if [ "$SAVE_TXT" = true ]; then
    echo "保存结果为txt文件: 是"
fi
if [ "$NOSAVE" = true ]; then
    echo "不保存图像/视频: 是"
fi
echo

# 确认开始测试
read -p "是否开始测试? [y/N] " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "已取消测试"
    exit 0
fi

# 开始测试
echo "开始测试..."
echo "执行命令: $COMMAND"
echo
$COMMAND

# 检查测试结果
if [ $? -eq 0 ]; then
    echo "测试完成!"
    echo "结果保存在: runs/detect/exp"
else
    echo "测试失败，请检查错误信息"
fi

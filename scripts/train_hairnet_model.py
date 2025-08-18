#!/usr/bin/env python

"""
使用YOLOv8训练发网检测模型

用法:
    python train_hairnet_model.py --data datasets/hairnet/data.yaml --epochs 100 --batch-size 16 --img-size 640
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="使用YOLOv8训练发网检测模型")
    parser.add_argument("--data", type=str, required=True, help="数据集配置文件路径(data.yaml)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，默认为100")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小，默认为16")
    parser.add_argument("--img-size", type=int, default=640, help="图像大小，默认为640")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/yolo/yolov8n.pt",
        help="初始权重，默认为models/yolo/yolov8n.pt",
    )
    parser.add_argument("--device", type=str, default="", help="训练设备，例如cuda:0或cpu")
    parser.add_argument(
        "--name", type=str, default="hairnet_model", help="实验名称，默认为hairnet_model"
    )
    parser.add_argument("--pretrained", action="store_true", help="使用预训练权重")
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    parser.add_argument(
        "--save-dir", type=str, default="./models", help="模型保存目录，默认为./models"
    )
    return parser.parse_args()


def check_dependencies():
    """检查依赖项"""
    try:
        import ultralytics
        from ultralytics import YOLO

        print(f"Ultralytics版本: {ultralytics.__version__}")
        return True
    except ImportError:
        print("错误: 未安装ultralytics库，请使用以下命令安装:")
        print("pip install ultralytics")
        return False


def check_data_yaml(yaml_path):
    """检查数据集配置文件"""
    if not os.path.exists(yaml_path):
        print(f"错误: 数据集配置文件不存在: {yaml_path}")
        return False

    try:
        import yaml

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        required_keys = ["path", "train", "val", "nc", "names"]
        for key in required_keys:
            if key not in data:
                print(f"错误: 数据集配置文件缺少必要的键: {key}")
                return False

        # 检查路径是否存在
        for key in ["train", "val"]:
            if not os.path.exists(data[key]):
                print(f"错误: {key}路径不存在: {data[key]}")
                return False

        print(f"数据集配置文件验证通过: {yaml_path}")
        print(f"类别: {data['names']}")
        return True
    except Exception as e:
        print(f"错误: 解析数据集配置文件时出错: {e}")
        return False


def train_yolov8(args):
    """使用YOLOv8训练模型"""
    from ultralytics import YOLO

    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型
    if args.pretrained:
        print(f"加载预训练模型: {args.weights}")
        model = YOLO(args.weights)
    else:
        # 从头开始训练
        print("从头开始训练YOLOv8模型")
        model = YOLO("yolov8n.yaml")

    # 设置训练参数
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.img_size,
        "project": args.save_dir,
        "name": args.name,
        "exist_ok": True,
    }

    # 添加设备参数（如果指定）
    if args.device:
        train_args["device"] = args.device

    # 添加恢复训练参数（如果指定）
    if args.resume:
        train_args["resume"] = True

    # 开始训练
    print("\n开始训练...")
    print(f"训练参数: {train_args}")
    model.train(**train_args)

    # 训练完成后，获取最佳模型路径
    best_model_path = os.path.join(args.save_dir, args.name, "weights", "best.pt")
    if os.path.exists(best_model_path):
        print(f"\n训练完成! 最佳模型保存在: {best_model_path}")

        # 复制最佳模型到models目录
        final_model_path = os.path.join(
            "models", "models/hairnet_detection/hairnet_detection.pt"
        )
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        import shutil

        shutil.copy2(best_model_path, final_model_path)
        print(f"最佳模型已复制到: {final_model_path}")

        return final_model_path
    else:
        print("\n训练完成，但未找到最佳模型文件")
        return None


def main():
    args = parse_args()

    # 检查依赖项
    if not check_dependencies():
        return 1

    # 检查数据集配置文件
    if not check_data_yaml(args.data):
        return 1

    # 训练模型
    try:
        model_path = train_yolov8(args)
        if model_path:
            print("\n模型训练成功!")
            print(f"模型路径: {model_path}")
            print("\n使用以下命令测试模型:")
            print(
                f"python test_hairnet_model.py --weights {model_path} --source path/to/test/image.jpg"
            )
            return 0
        else:
            print("\n模型训练失败")
            return 1
    except Exception as e:
        print(f"\n训练过程中出错: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

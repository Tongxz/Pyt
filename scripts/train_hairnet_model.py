#!/usr/bin/env python3
"""
发网检测模型训练脚本

使用Roboflow数据集训练YOLOv8模型进行发网检测
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hairnet_training")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="发网检测模型训练脚本")
    parser.add_argument(
        "--data", type=str, required=True, help="Roboflow数据集路径或YAML配置文件路径"
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--img-size", type=int, default=640, help="图像尺寸")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="基础模型，可选：yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="训练设备，可选：cpu, cuda, auto"
    )
    parser.add_argument("--name", type=str, default="hairnet_model", help="实验名称")
    parser.add_argument("--pretrained", action="store_true", help="使用预训练权重")
    parser.add_argument("--resume", action="store_true", help="从上次检查点恢复训练")
    parser.add_argument("--save-dir", type=str, default="./models", help="模型保存目录")

    return parser.parse_args()


def create_data_yaml(data_path):
    """创建数据集YAML配置文件"""
    # 检查数据集路径是否已经是YAML文件
    if data_path.endswith(".yaml"):
        logger.info(f"使用现有的YAML配置文件: {data_path}")
        return data_path

    # 检查数据集路径是否存在
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")

    # 检查数据集结构
    train_dir = data_dir / "train" / "images"
    val_dir = data_dir / "valid" / "images"
    test_dir = data_dir / "test" / "images"

    if not train_dir.exists():
        raise FileNotFoundError(f"训练集路径不存在: {train_dir}")

    # 创建YAML配置
    yaml_path = data_dir / "data.yaml"

    # 检查是否已存在data.yaml文件
    if yaml_path.exists():
        logger.info(f"使用现有的数据集YAML配置: {yaml_path}")
        return str(yaml_path)

    # 从标签目录获取类别名称
    classes = []
    labels_dir = data_dir / "train" / "labels"
    if labels_dir.exists():
        # 尝试从目录中的第一个标签文件获取类别数量
        label_files = list(labels_dir.glob("*.txt"))
        if label_files:
            with open(label_files[0], "r") as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                    classes.extend(["class_" + str(i) for i in range(class_id + 1)])

    # 如果无法从标签文件获取类别，则假设只有一个类别：hairnet
    if not classes:
        classes = ["hairnet"]

    # 创建YAML配置
    data_yaml = {
        "path": str(data_dir),
        "train": str(train_dir),
        "val": str(val_dir) if val_dir.exists() else str(train_dir),
        "test": str(test_dir)
        if test_dir.exists()
        else str(val_dir)
        if val_dir.exists()
        else str(train_dir),
        "nc": len(classes),
        "names": classes,
    }

    # 写入YAML文件
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    logger.info(f"创建数据集YAML配置: {yaml_path}")
    logger.info(f"类别: {classes}")

    return str(yaml_path)


def train_model(args):
    """训练YOLOv8模型"""
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 准备数据集配置
    data_yaml = create_data_yaml(args.data)

    # 选择设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"使用设备: {device}")

    # 加载模型
    if args.pretrained:
        logger.info(f"加载预训练模型: {args.model}")
        model = YOLO(args.model)
    else:
        # 从头开始训练，需要指定YAML配置
        logger.info("从头开始训练模型")
        model_yaml = (
            f'yolov8{args.model[6] if args.model.startswith("yolov8") else "n"}.yaml'
        )
        model = YOLO(model_yaml)

    # 训练模型
    logger.info("开始训练...")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=device,
        name=args.name,
        resume=args.resume,
        project=str(save_dir),
        exist_ok=True,
    )

    # 保存最终模型
    final_model_path = save_dir / args.name / "weights" / "best.pt"
    hairnet_model_path = save_dir / "hairnet_detector.pt"

    if final_model_path.exists():
        # 复制最佳模型到项目模型目录
        import shutil

        shutil.copy(final_model_path, hairnet_model_path)
        logger.info(f"最佳模型已保存至: {hairnet_model_path}")

    # 验证模型
    logger.info("验证模型...")
    model.val()

    return results


def main():
    """主函数"""
    args = parse_args()

    logger.info("=" * 50)
    logger.info("发网检测模型训练开始")
    logger.info(f"数据集路径: {args.data}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"图像尺寸: {args.img_size}")
    logger.info(f"基础模型: {args.model}")
    logger.info(f"实验名称: {args.name}")
    logger.info("=" * 50)

    try:
        results = train_model(args)
        logger.info("训练完成!")
        logger.info(f"最终mAP: {results.maps}")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

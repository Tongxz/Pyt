#!/usr/bin/env python

"""
处理从Roboflow下载的数据集，并将其转换为YOLOv8格式

用法:
    python prepare_roboflow_dataset.py --input /path/to/roboflow_dataset.zip --output datasets/hairnet
    或
    python prepare_roboflow_dataset.py --input /path/to/extracted_roboflow_folder --output datasets/hairnet
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="处理Roboflow数据集并准备用于YOLOv8训练")
    parser.add_argument(
        "--input", type=str, required=True, help="Roboflow数据集的ZIP文件路径或已解压的目录路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/hairnet",
        help="输出目录，默认为datasets/hairnet",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="yolov8",
        choices=["yolov8", "yolov5", "coco"],
        help="数据集格式，默认为yolov8",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "roboflow"],
        help="数据集分割方式，auto使用Roboflow的分割，roboflow使用Roboflow的train/valid/test分割",
    )
    parser.add_argument("--force", action="store_true", help="强制覆盖输出目录")
    return parser.parse_args()


def extract_zip(zip_path, output_dir):
    """解压ZIP文件到指定目录"""
    print(f"正在解压 {zip_path} 到 {output_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"解压完成")

    # 返回解压后的根目录
    extracted_dirs = [
        d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
    ]
    if len(extracted_dirs) == 1:
        return os.path.join(output_dir, extracted_dirs[0])
    return output_dir


def detect_roboflow_format(input_dir):
    """检测Roboflow数据集的格式"""
    # 检查是否有data.yaml文件
    yaml_files = [
        f for f in os.listdir(input_dir) if f.endswith(".yaml") or f.endswith(".yml")
    ]

    if yaml_files:
        for yaml_file in yaml_files:
            with open(os.path.join(input_dir, yaml_file), "r") as f:
                try:
                    data = yaml.safe_load(f)
                    if "names" in data:
                        print(f"检测到YOLO格式数据集，类别: {data['names']}")
                        return "yolo", data
                except Exception as e:
                    print(f"解析YAML文件时出错: {e}")

    # 检查目录结构
    if os.path.exists(os.path.join(input_dir, "train")) and os.path.exists(
        os.path.join(input_dir, "valid")
    ):
        if os.path.exists(
            os.path.join(input_dir, "train", "images")
        ) and os.path.exists(os.path.join(input_dir, "train", "labels")):
            print("检测到YOLO格式数据集目录结构")
            return "yolo", None

    if (
        os.path.exists(os.path.join(input_dir, "train"))
        and os.path.exists(os.path.join(input_dir, "valid"))
        and os.path.exists(os.path.join(input_dir, "_annotations.coco.json"))
    ):
        print("检测到COCO格式数据集")
        return "coco", None

    print("无法检测到数据集格式，将使用默认YOLO格式处理")
    return "unknown", None


def organize_yolo_dataset(input_dir, output_dir, split="auto"):
    """组织YOLO格式的数据集"""
    # 创建输出目录结构
    for split_name in ["train", "valid", "test"]:
        for subdir in ["images", "labels"]:
            os.makedirs(os.path.join(output_dir, split_name, subdir), exist_ok=True)

    # 检查输入目录结构
    if split == "auto":
        # 自动检测目录结构
        if os.path.exists(os.path.join(input_dir, "train")) and os.path.exists(
            os.path.join(input_dir, "valid")
        ):
            split = "roboflow"
        else:
            # 假设所有图像和标签都在同一目录
            split = "single"

    if split == "roboflow":
        # Roboflow的train/valid/test分割
        for split_name in ["train", "valid", "test"]:
            if not os.path.exists(os.path.join(input_dir, split_name)):
                if split_name == "test":
                    print(f"警告: 未找到测试集目录 {split_name}")
                    continue
                else:
                    print(f"错误: 未找到必要的数据集目录 {split_name}")
                    return False

            # 复制图像
            src_img_dir = os.path.join(input_dir, split_name, "images")
            if os.path.exists(src_img_dir):
                dst_img_dir = os.path.join(output_dir, split_name, "images")
                for img in os.listdir(src_img_dir):
                    if img.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        shutil.copy2(
                            os.path.join(src_img_dir, img),
                            os.path.join(dst_img_dir, img),
                        )

            # 复制标签
            src_lbl_dir = os.path.join(input_dir, split_name, "labels")
            if os.path.exists(src_lbl_dir):
                dst_lbl_dir = os.path.join(output_dir, split_name, "labels")
                for lbl in os.listdir(src_lbl_dir):
                    if lbl.endswith(".txt"):
                        shutil.copy2(
                            os.path.join(src_lbl_dir, lbl),
                            os.path.join(dst_lbl_dir, lbl),
                        )
    else:
        # 单一目录，需要手动分割
        print("错误: 不支持单一目录的数据集，请使用已分割的Roboflow数据集")
        return False

    return True


def create_data_yaml(output_dir, class_names=None):
    """创建或更新data.yaml文件"""
    yaml_path = os.path.join(output_dir, "data.yaml")

    # 如果没有提供类别名称，尝试从现有的yaml文件中读取
    if class_names is None:
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                class_names = data.get("names", ["hairnet"])
        else:
            class_names = ["hairnet"]

    # 准备YAML数据
    data = {
        "path": os.path.abspath(output_dir),
        "train": os.path.join(os.path.abspath(output_dir), "train", "images"),
        "val": os.path.join(os.path.abspath(output_dir), "valid", "images"),
        "test": os.path.join(os.path.abspath(output_dir), "test", "images"),
        "nc": len(class_names),
        "names": class_names,
    }

    # 写入YAML文件
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"已创建数据集配置文件: {yaml_path}")
    return yaml_path


def validate_dataset(output_dir):
    """验证数据集结构和内容"""
    # 检查目录结构
    for split_name in ["train", "valid"]:
        for subdir in ["images", "labels"]:
            dir_path = os.path.join(output_dir, split_name, subdir)
            if not os.path.exists(dir_path):
                print(f"错误: 缺少目录 {dir_path}")
                return False

    # 检查图像和标签数量
    for split_name in ["train", "valid", "test"]:
        if not os.path.exists(os.path.join(output_dir, split_name)):
            if split_name == "test":
                continue
            else:
                print(f"错误: 缺少必要的数据集目录 {split_name}")
                return False

        img_dir = os.path.join(output_dir, split_name, "images")
        lbl_dir = os.path.join(output_dir, split_name, "labels")

        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            img_count = len(
                [
                    f
                    for f in os.listdir(img_dir)
                    if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ]
            )
            lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith(".txt")])

            print(f"{split_name}集: {img_count}张图像, {lbl_count}个标签文件")

            if img_count == 0:
                print(f"警告: {split_name}集没有图像")
                if split_name != "test":
                    return False

            if lbl_count == 0:
                print(f"警告: {split_name}集没有标签文件")
                if split_name != "test":
                    return False

            if img_count != lbl_count and split_name != "test":
                print(f"警告: {split_name}集的图像数量({img_count})与标签数量({lbl_count})不匹配")

    return True


def main():
    args = parse_args()

    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在: {args.input}")
        return 1

    # 检查输出目录
    if os.path.exists(args.output) and os.listdir(args.output) and not args.force:
        print(f"错误: 输出目录已存在且不为空: {args.output}")
        print("使用 --force 参数覆盖现有目录")
        return 1

    # 创建临时目录
    temp_dir = os.path.join(os.path.dirname(args.output), "temp_roboflow")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 处理输入
        if args.input.endswith(".zip"):
            input_dir = extract_zip(args.input, temp_dir)
        else:
            input_dir = args.input

        # 检测数据集格式
        format_type, yaml_data = detect_roboflow_format(input_dir)

        # 创建输出目录
        os.makedirs(args.output, exist_ok=True)

        # 根据格式组织数据集
        if format_type in ["yolo", "unknown"]:
            success = organize_yolo_dataset(input_dir, args.output, args.split)
            if not success:
                print("错误: 组织数据集失败")
                return 1
        else:
            print(f"错误: 不支持的数据集格式: {format_type}")
            return 1

        # 创建data.yaml
        class_names = yaml_data.get("names") if yaml_data else None
        yaml_path = create_data_yaml(args.output, class_names)

        # 验证数据集
        if validate_dataset(args.output):
            print("\n数据集准备完成!")
            print(f"数据集路径: {os.path.abspath(args.output)}")
            print(f"配置文件: {yaml_path}")
            print("\n使用以下命令训练YOLOv8模型:")
            print(
                f"python train_hairnet_model.py --data {yaml_path} --epochs 100 --batch-size 16 --img-size 640"
            )
        else:
            print("\n警告: 数据集验证失败，请检查数据集结构和内容")

    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())

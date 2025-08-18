#!/usr/bin/env python3
"""
增加数据集脚本
用于向现有的发网检测数据集添加新的训练数据

功能:
1. 添加新的图片和标注文件
2. 验证标注格式
3. 更新数据集统计信息
4. 重新生成缓存文件
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


class DatasetExpander:
    def __init__(self, dataset_root: str = "datasets/hairnet"):
        self.dataset_root = Path(dataset_root)
        self.data_yaml_path = self.dataset_root / "data.yaml"

        # 加载现有数据集配置
        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.classes = self.config["names"]
        self.num_classes = len(self.classes)

        print(f"当前数据集类别: {self.classes}")
        print(f"类别数量: {self.num_classes}")

    def validate_annotation_format(self, label_file: str) -> bool:
        """
        验证YOLO格式标注文件
        格式: class_id center_x center_y width height (归一化坐标)
        """
        try:
            with open(label_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    print(f"错误: {label_file} 格式不正确，应为5个值")
                    return False

                class_id = int(parts[0])
                if class_id >= self.num_classes:
                    print(f"错误: {label_file} 类别ID {class_id} 超出范围")
                    return False

                # 检查坐标是否在0-1范围内
                coords = [float(x) for x in parts[1:]]
                if not all(0 <= coord <= 1 for coord in coords):
                    print(f"错误: {label_file} 坐标不在0-1范围内")
                    return False

            return True
        except Exception as e:
            print(f"验证标注文件 {label_file} 时出错: {e}")
            return False

    def add_images_and_labels(
        self,
        source_images_dir: str,
        source_labels_dir: str,
        target_split: str = "train",
    ) -> bool:
        """
        添加图片和标注文件到指定的数据集分割

        Args:
            source_images_dir: 源图片目录
            source_labels_dir: 源标注目录
            target_split: 目标分割 (train/valid/test)
        """
        source_images = Path(source_images_dir)
        source_labels = Path(source_labels_dir)

        if not source_images.exists():
            print(f"错误: 源图片目录不存在: {source_images}")
            return False

        if not source_labels.exists():
            print(f"错误: 源标注目录不存在: {source_labels}")
            return False

        # 目标目录
        target_images_dir = self.dataset_root / target_split / "images"
        target_labels_dir = self.dataset_root / target_split / "labels"

        # 创建目标目录
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_labels_dir.mkdir(parents=True, exist_ok=True)

        # 获取图片文件列表
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_images.glob(f"*{ext}"))
            image_files.extend(source_images.glob(f"*{ext.upper()}"))

        added_count = 0
        skipped_count = 0

        for image_file in image_files:
            # 对应的标注文件
            label_file = source_labels / f"{image_file.stem}.txt"

            if not label_file.exists():
                print(f"警告: 图片 {image_file.name} 没有对应的标注文件，跳过")
                skipped_count += 1
                continue

            # 验证标注格式
            if not self.validate_annotation_format(str(label_file)):
                print(f"警告: 标注文件 {label_file.name} 格式不正确，跳过")
                skipped_count += 1
                continue

            # 复制文件
            target_image = target_images_dir / image_file.name
            target_label = target_labels_dir / label_file.name

            # 检查是否已存在
            if target_image.exists():
                print(f"警告: 图片 {image_file.name} 已存在，跳过")
                skipped_count += 1
                continue

            try:
                shutil.copy2(image_file, target_image)
                shutil.copy2(label_file, target_label)
                added_count += 1
                print(f"添加: {image_file.name}")
            except Exception as e:
                print(f"复制文件时出错: {e}")
                skipped_count += 1

        print(f"\n添加完成:")
        print(f"成功添加: {added_count} 个样本")
        print(f"跳过: {skipped_count} 个样本")

        # 删除缓存文件，强制重新生成
        cache_file = target_labels_dir / "labels.cache"
        if cache_file.exists():
            cache_file.unlink()
            print(f"已删除缓存文件: {cache_file}")

        return added_count > 0

    def convert_labelme_to_yolo(
        self, labelme_json_dir: str, output_labels_dir: str
    ) -> bool:
        """
        将LabelMe JSON格式转换为YOLO格式
        """
        labelme_dir = Path(labelme_json_dir)
        output_dir = Path(output_labels_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(labelme_dir.glob("*.json"))
        if not json_files:
            print(f"在 {labelme_dir} 中没有找到JSON文件")
            return False

        converted_count = 0

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                image_width = data["imageWidth"]
                image_height = data["imageHeight"]

                # 输出标注文件
                output_file = output_dir / f"{json_file.stem}.txt"

                with open(output_file, "w") as f:
                    for shape in data["shapes"]:
                        label = shape["label"].lower()

                        # 映射类别名称到ID
                        if label in self.classes:
                            class_id = self.classes.index(label)
                        else:
                            print(f"警告: 未知类别 '{label}' 在文件 {json_file.name}")
                            continue

                        # 获取边界框坐标
                        points = shape["points"]
                        if len(points) != 2:  # 矩形应该有2个点
                            print(f"警告: 形状不是矩形 在文件 {json_file.name}")
                            continue

                        x1, y1 = points[0]
                        x2, y2 = points[1]

                        # 确保坐标顺序正确
                        x_min, x_max = min(x1, x2), max(x1, x2)
                        y_min, y_max = min(y1, y2), max(y1, y2)

                        # 转换为YOLO格式 (归一化的中心坐标和宽高)
                        center_x = (x_min + x_max) / 2 / image_width
                        center_y = (y_min + y_max) / 2 / image_height
                        width = (x_max - x_min) / image_width
                        height = (y_max - y_min) / image_height

                        f.write(
                            f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                        )

                converted_count += 1
                print(f"转换: {json_file.name} -> {output_file.name}")

            except Exception as e:
                print(f"转换文件 {json_file.name} 时出错: {e}")

        print(f"\n转换完成: {converted_count} 个文件")
        return converted_count > 0

    def get_dataset_statistics(self) -> Dict:
        """
        获取数据集统计信息
        """
        stats = {
            "train": {"images": 0, "labels": 0, "objects": 0},
            "valid": {"images": 0, "labels": 0, "objects": 0},
            "test": {"images": 0, "labels": 0, "objects": 0},
        }

        for split in ["train", "valid", "test"]:
            images_dir = self.dataset_root / split / "images"
            labels_dir = self.dataset_root / split / "labels"

            if images_dir.exists():
                image_files = list(images_dir.glob("*"))
                stats[split]["images"] = len([f for f in image_files if f.is_file()])

            if labels_dir.exists():
                label_files = list(labels_dir.glob("*.txt"))
                label_files = [f for f in label_files if f.name != "labels.cache"]
                stats[split]["labels"] = len(label_files)

                # 统计目标数量
                object_count = 0
                for label_file in label_files:
                    try:
                        with open(label_file, "r") as f:
                            lines = f.readlines()
                        object_count += len([line for line in lines if line.strip()])
                    except:
                        pass
                stats[split]["objects"] = object_count

        return stats

    def print_statistics(self):
        """
        打印数据集统计信息
        """
        stats = self.get_dataset_statistics()

        print("\n=== 数据集统计信息 ===")
        print(f"类别: {self.classes}")
        print(f"类别数量: {self.num_classes}")
        print()

        total_images = 0
        total_objects = 0

        for split in ["train", "valid", "test"]:
            images = stats[split]["images"]
            labels = stats[split]["labels"]
            objects = stats[split]["objects"]

            print(f"{split.upper()}:")
            print(f"  图片: {images}")
            print(f"  标注: {labels}")
            print(f"  目标: {objects}")
            print()

            total_images += images
            total_objects += objects

        print(f"总计:")
        print(f"  图片: {total_images}")
        print(f"  目标: {total_objects}")


def main():
    """
    主函数 - 提供交互式界面
    """
    expander = DatasetExpander()

    while True:
        print("\n=== 数据集扩展工具 ===")
        print("1. 查看当前数据集统计")
        print("2. 添加新的图片和标注 (YOLO格式)")
        print("3. 转换LabelMe JSON到YOLO格式")
        print("4. 退出")

        choice = input("\n请选择操作 (1-4): ").strip()

        if choice == "1":
            expander.print_statistics()

        elif choice == "2":
            print("\n添加新的训练数据")
            images_dir = input("请输入图片目录路径: ").strip()
            labels_dir = input("请输入标注目录路径: ").strip()

            split_choice = input("添加到哪个数据集? (train/valid/test) [默认: train]: ").strip()
            if not split_choice:
                split_choice = "train"

            if split_choice not in ["train", "valid", "test"]:
                print("错误: 无效的数据集分割")
                continue

            success = expander.add_images_and_labels(
                images_dir, labels_dir, split_choice
            )
            if success:
                print("\n数据添加成功！")
                expander.print_statistics()
            else:
                print("\n数据添加失败！")

        elif choice == "3":
            print("\n转换LabelMe JSON格式")
            json_dir = input("请输入LabelMe JSON目录路径: ").strip()
            output_dir = input("请输入输出标注目录路径: ").strip()

            success = expander.convert_labelme_to_yolo(json_dir, output_dir)
            if success:
                print("\n格式转换成功！")
                print("现在可以使用选项2将转换后的数据添加到数据集中")
            else:
                print("\n格式转换失败！")

        elif choice == "4":
            print("退出程序")
            break

        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    main()

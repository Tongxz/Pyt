#!/usr/bin/env python3
"""
测试文件整理脚本

将根目录下的测试文件整理到合适的位置
"""

import glob
import os
import re
import shutil
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 测试目录
TESTS_DIR = ROOT_DIR / "tests"

# 单元测试目录
UNIT_TESTS_DIR = TESTS_DIR / "unit"

# 集成测试目录
INTEGRATION_TESTS_DIR = TESTS_DIR / "integration"

# 测试数据目录
FIXTURES_DIR = TESTS_DIR / "fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
VIDEOS_DIR = FIXTURES_DIR / "videos"
PERSON_IMAGES_DIR = IMAGES_DIR / "person"
HAIRNET_IMAGES_DIR = IMAGES_DIR / "hairnet"


# 确保目录存在
def ensure_dirs():
    """确保所有目录存在"""
    for dir_path in [
        UNIT_TESTS_DIR,
        INTEGRATION_TESTS_DIR,
        FIXTURES_DIR,
        IMAGES_DIR,
        VIDEOS_DIR,
        PERSON_IMAGES_DIR,
        HAIRNET_IMAGES_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


# 移动测试文件
def move_test_files():
    """移动测试文件到合适的位置"""
    # 获取根目录下的所有测试文件
    test_files = [f for f in ROOT_DIR.glob("test_*.py") if f.is_file()]

    # 移动文件计数
    moved_count = 0

    for test_file in test_files:
        # 确定目标目录
        if is_unit_test(test_file):
            target_dir = UNIT_TESTS_DIR
        elif is_integration_test(test_file):
            target_dir = INTEGRATION_TESTS_DIR
        else:
            # 默认为单元测试
            target_dir = UNIT_TESTS_DIR

        # 目标文件路径
        target_path = target_dir / test_file.name

        # 如果目标文件已存在，添加后缀
        if target_path.exists():
            # 检查内容是否相同
            if are_files_identical(test_file, target_path):
                print(f"跳过相同文件: {test_file.name}")
                continue
            else:
                # 添加后缀
                base_name = target_path.stem
                extension = target_path.suffix
                counter = 1
                while target_path.exists():
                    target_path = target_dir / f"{base_name}_{counter}{extension}"
                    counter += 1

        # 复制文件
        shutil.copy2(test_file, target_path)
        print(f"已复制: {test_file} -> {target_path}")
        moved_count += 1

    return moved_count


# 移动测试图像
def move_test_images():
    """移动测试图像到测试数据目录"""
    # 图像文件扩展名
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    # 获取根目录下的所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(ROOT_DIR.glob(f"*{ext}"))

    # 移动文件计数
    moved_count = 0

    for image_file in image_files:
        # 确定目标目录
        if is_person_image(image_file):
            target_dir = PERSON_IMAGES_DIR
        elif is_hairnet_image(image_file):
            target_dir = HAIRNET_IMAGES_DIR
        else:
            # 默认为人物图像
            target_dir = PERSON_IMAGES_DIR

        # 目标文件路径
        target_path = target_dir / image_file.name

        # 如果目标文件已存在，添加后缀
        if target_path.exists():
            # 检查内容是否相同
            if are_files_identical(image_file, target_path):
                print(f"跳过相同图像: {image_file.name}")
                continue
            else:
                # 添加后缀
                base_name = target_path.stem
                extension = target_path.suffix
                counter = 1
                while target_path.exists():
                    target_path = target_dir / f"{base_name}_{counter}{extension}"
                    counter += 1

        # 复制文件
        shutil.copy2(image_file, target_path)
        print(f"已复制: {image_file} -> {target_path}")
        moved_count += 1

    return moved_count


# 判断文件类型的辅助函数
def is_unit_test(file_path):
    """判断是否为单元测试文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # 单元测试通常包含unittest或pytest的特定模式
        return (
            "unittest" in content
            or "pytest" in content
            or "TestCase" in content
            or re.search(r"def test_\w+\(.*\):", content) is not None
        ) and not is_integration_test(file_path)


def is_integration_test(file_path):
    """判断是否为集成测试文件"""
    # 基于文件名判断
    if "integration" in file_path.name or "api" in file_path.name:
        return True

    # 基于文件内容判断
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # 集成测试通常包含API调用、HTTP请求等
        return (
            "requests." in content
            or "http" in content.lower()
            or "api" in content.lower()
            or "integration" in content.lower()
        )


def is_person_image(file_path):
    """判断是否为人物图像"""
    return (
        "person" in file_path.name.lower() and not "hairnet" in file_path.name.lower()
    )


def is_hairnet_image(file_path):
    """判断是否为发网图像"""
    return "hairnet" in file_path.name.lower()


def are_files_identical(file1, file2):
    """判断两个文件内容是否相同"""
    try:
        with open(file1, "rb") as f1, open(file2, "rb") as f2:
            return f1.read() == f2.read()
    except Exception:
        return False


# 主函数
def main():
    """主函数"""
    print("=== 测试文件整理 ===")

    # 确保目录存在
    ensure_dirs()
    print("目录结构已创建")

    # 移动测试文件
    moved_files = move_test_files()
    print(f"已整理 {moved_files} 个测试文件")

    # 移动测试图像
    moved_images = move_test_images()
    print(f"已整理 {moved_images} 个测试图像")

    print("\n测试文件整理完成！")
    print("注意: 原始文件仍保留在根目录，请手动删除不需要的文件")


if __name__ == "__main__":
    main()

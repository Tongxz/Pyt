#!/usr/bin/env python3
"""
清理项目根目录

这个脚本用于：
1. 删除已经被整理到tests目录中的根目录测试文件和测试图像
2. 将不必要的脚本文件从根目录移动到scripts目录
"""

import os
import shutil
import sys
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent
# 获取scripts目录
SCRIPTS_DIR = ROOT_DIR / "scripts"

# 要删除的根目录测试文件列表
ROOT_TEST_FILES = [
    "test_api.py",
    "test_api_integration.py",
    "test_detailed_hairnet_analysis.py",
    "test_detect_hairnet.py",
    "test_dual_channel_hairnet.py",
    "test_hairnet_detection.py",
    "test_integrated_hairnet_detection.py",
    "test_light_blue_hairnet.py",
    "test_real_hairnet_image.py",
    "test_realtime_detection.py",
    "test_roi_visualization.py",
    "test_sprint2_api.py",
    "test_threshold_adjustment.py",
    "test_with_hairnet.py",
]

# 要删除的根目录测试图像文件列表
ROOT_TEST_IMAGES = [
    "real_hairnet_test.jpg",
    "enhanced_roi_analysis.png",
    "test_person.png",
    "factory_test.jpg",
    "test_image.jpg",
    "test_person_roi_analysis.png",
    "test_light_blue_hairnet.jpg",
    "improved_head_roi_comparison_person_1.png",
    "improved_head_roi_comparison_person_2.png",
    "improved_head_roi_comparison_person_3.png",
    "realistic_test_image.jpg",
    "realistic_test_preview.png",
]

# 要移动到scripts目录的根目录脚本文件列表
ROOT_SCRIPT_FILES = [
    "analyze_detection_parameters.py",
    "create_realistic_test.py",
    "debug_detection_parameters.py",
    "detection_config_optimized.py",
    "enhanced_roi_visualizer.py",
    "improved_head_roi.py",
    "view_enhanced_results.py",
    "view_improved_roi.py",
    "view_roi_results.py",
    "visualize_roi.py",
]

# 要移动到data目录的数据文件列表
ROOT_DATA_FILES = [
    "detection_results.db",
]


def main():
    """清理根目录下的过期测试文件、测试图像、不必要的脚本文件和数据文件"""
    print("开始清理过期测试文件和测试图像...")

    # 统计信息
    deleted_files_count = 0
    skipped_files_count = 0
    error_files_count = 0

    deleted_images_count = 0
    skipped_images_count = 0
    error_images_count = 0

    moved_scripts_count = 0
    skipped_scripts_count = 0
    error_scripts_count = 0

    moved_data_count = 0
    skipped_data_count = 0
    error_data_count = 0

    # 第一部分：删除测试文件
    print("\n1. 清理测试文件:")
    for filename in ROOT_TEST_FILES:
        file_path = ROOT_DIR / filename

        if file_path.exists():
            try:
                # 删除文件
                file_path.unlink()
                print(f"已删除: {filename}")
                deleted_files_count += 1
            except Exception as e:
                print(f"删除失败: {filename} - {str(e)}")
                error_files_count += 1
        else:
            print(f"跳过: {filename} (文件不存在)")
            skipped_files_count += 1

    # 第二部分：删除测试图像
    print("\n2. 清理测试图像:")
    for imagename in ROOT_TEST_IMAGES:
        image_path = ROOT_DIR / imagename

        if image_path.exists():
            try:
                # 删除图像文件
                image_path.unlink()
                print(f"已删除: {imagename}")
                deleted_images_count += 1
            except Exception as e:
                print(f"删除失败: {imagename} - {str(e)}")
                error_images_count += 1
        else:
            print(f"跳过: {imagename} (文件不存在)")
            skipped_images_count += 1

    # 第三部分：移动脚本文件到scripts目录
    print("\n3. 移动脚本文件到scripts目录:")
    for scriptname in ROOT_SCRIPT_FILES:
        script_path = ROOT_DIR / scriptname
        target_path = SCRIPTS_DIR / scriptname

        if script_path.exists():
            try:
                # 如果目标文件已存在，先备份
                if target_path.exists():
                    backup_path = target_path.with_suffix(".py.bak")
                    shutil.move(str(target_path), str(backup_path))
                    print(f"已备份: {target_path.name} -> {backup_path.name}")

                # 移动文件
                shutil.move(str(script_path), str(target_path))
                print(f"已移动: {scriptname} -> scripts/{scriptname}")
                moved_scripts_count += 1
            except Exception as e:
                print(f"移动失败: {scriptname} - {str(e)}")
                error_scripts_count += 1
        else:
            print(f"跳过: {scriptname} (文件不存在)")
            skipped_scripts_count += 1

    # 第四部分：移动数据文件到data目录
    print("\n4. 移动数据文件到data目录:")
    # 确保data目录存在
    data_dir = ROOT_DIR / "data"
    if not data_dir.exists():
        data_dir.mkdir()
        print(f"已创建data目录: {data_dir}")

    for dataname in ROOT_DATA_FILES:
        data_path = ROOT_DIR / dataname
        target_path = data_dir / dataname

        if data_path.exists():
            try:
                # 如果目标文件已存在，先备份
                if target_path.exists():
                    backup_path = target_path.with_suffix(".db.bak")
                    shutil.move(str(target_path), str(backup_path))
                    print(f"已备份: {target_path.name} -> {backup_path.name}")

                # 移动文件
                shutil.move(str(data_path), str(target_path))
                print(f"已移动: {dataname} -> data/{dataname}")
                moved_data_count += 1
            except Exception as e:
                print(f"移动失败: {dataname} - {str(e)}")
                error_data_count += 1
        else:
            print(f"跳过: {dataname} (文件不存在)")
            skipped_data_count += 1

    # 打印统计信息
    print("\n清理完成!")
    print(
        f"测试文件: 删除={deleted_files_count}, 跳过={skipped_files_count}, 错误={error_files_count}"
    )
    print(
        f"测试图像: 删除={deleted_images_count}, 跳过={skipped_images_count}, 错误={error_images_count}"
    )
    print(
        f"脚本文件: 移动={moved_scripts_count}, 跳过={skipped_scripts_count}, 错误={error_scripts_count}"
    )
    print(
        f"数据文件: 移动={moved_data_count}, 跳过={skipped_data_count}, 错误={error_data_count}"
    )
    print(
        f"总计: 处理={deleted_files_count + deleted_images_count + moved_scripts_count + moved_data_count}, "
        f"跳过={skipped_files_count + skipped_images_count + skipped_scripts_count + skipped_data_count}, "
        f"错误={error_files_count + error_images_count + error_scripts_count + error_data_count}"
    )


if __name__ == "__main__":
    main()

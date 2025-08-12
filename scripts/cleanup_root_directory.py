#!/usr/bin/env python3
"""
清理项目根目录垃圾文件和测试文件

这个脚本用于：
1. 删除根目录下的临时测试文件
2. 删除调试文件
3. 删除临时HTML文件
4. 删除日志文件
5. 删除系统临时文件（如.DS_Store）
6. 保留重要的配置文件和项目文件
"""

import os
import shutil
import sys
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent

# 要删除的测试文件列表（根目录下的临时测试文件）
TEST_FILES_TO_DELETE = [
    "test_yolo_model_info.py",
    "test_updated_detector.py",
    "test_real_image.py",
    "test_performance_optimization.py",
    "test_optimized_pipeline.py",
    "test_optimization_summary.py",
    "test_image_detection.py",
    "test_head_region_extraction.py",
    "test_hairnet_detector_only.py",
    "test_fixed_detection.py",
    "test_comprehensive_detection.py",
    "test_behavior_detection.py",
    "test_api.py",
    "test_hairnet_model.py",
]

# 要删除的调试文件列表
DEBUG_FILES_TO_DELETE = [
    "debug_human_detector.py",
    "debug_api_response.py",
]

# 要删除的临时HTML文件列表
HTML_FILES_TO_DELETE = [
    "test_upload.html",
]

# 要删除的日志文件列表
LOG_FILES_TO_DELETE = [
    "frontend.log",
    "backend.log",
]

# 要删除的系统临时文件列表
SYSTEM_TEMP_FILES_TO_DELETE = [
    ".DS_Store",
]

# 要保留的重要文件（不删除）
IMPORTANT_FILES_TO_KEEP = [
    ".gitignore",
    ".dockerignore",
    ".flake8",
    ".pre-commit-config.yaml",
    ".python-version",
    "main.py",
    "README.md",
    "requirements.txt",
    "requirements.dev.txt",
    "pyproject.toml",
    "pytest.ini",
    "mypy.ini",
    "Makefile",
    "LICENSE",
    "CONTRIBUTING.md",
    "Dockerfile",
    "Dockerfile.dev",
    "docker-compose.yml",
]


def delete_files(file_list, category_name):
    """删除指定的文件列表"""
    print(f"\n{category_name}:")
    deleted_count = 0
    skipped_count = 0
    error_count = 0

    for filename in file_list:
        file_path = ROOT_DIR / filename

        if file_path.exists():
            try:
                file_path.unlink()
                print(f"  ✓ 已删除: {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"  ✗ 删除失败: {filename} - {str(e)}")
                error_count += 1
        else:
            print(f"  - 跳过: {filename} (文件不存在)")
            skipped_count += 1

    print(f"  统计: 删除={deleted_count}, 跳过={skipped_count}, 错误={error_count}")
    return deleted_count, skipped_count, error_count


def confirm_deletion():
    """确认是否执行删除操作"""
    print("\n=== 项目根目录清理工具 ===")
    print("\n将要删除以下类型的文件:")
    print(f"- 测试文件: {len(TEST_FILES_TO_DELETE)} 个")
    print(f"- 调试文件: {len(DEBUG_FILES_TO_DELETE)} 个")
    print(f"- 临时HTML文件: {len(HTML_FILES_TO_DELETE)} 个")
    print(f"- 日志文件: {len(LOG_FILES_TO_DELETE)} 个")
    print(f"- 系统临时文件: {len(SYSTEM_TEMP_FILES_TO_DELETE)} 个")

    print("\n重要文件将被保留（如配置文件、README等）")

    response = input("\n确认执行清理操作？(y/N): ").strip().lower()
    return response in ["y", "yes"]


def main():
    """主函数"""
    if not confirm_deletion():
        print("操作已取消")
        return

    print("\n开始清理根目录...")

    # 统计总数
    total_deleted = 0
    total_skipped = 0
    total_errors = 0

    # 删除各类文件
    deleted, skipped, errors = delete_files(TEST_FILES_TO_DELETE, "1. 清理测试文件")
    total_deleted += deleted
    total_skipped += skipped
    total_errors += errors

    deleted, skipped, errors = delete_files(DEBUG_FILES_TO_DELETE, "2. 清理调试文件")
    total_deleted += deleted
    total_skipped += skipped
    total_errors += errors

    deleted, skipped, errors = delete_files(HTML_FILES_TO_DELETE, "3. 清理临时HTML文件")
    total_deleted += deleted
    total_skipped += skipped
    total_errors += errors

    deleted, skipped, errors = delete_files(LOG_FILES_TO_DELETE, "4. 清理日志文件")
    total_deleted += deleted
    total_skipped += skipped
    total_errors += errors

    deleted, skipped, errors = delete_files(SYSTEM_TEMP_FILES_TO_DELETE, "5. 清理系统临时文件")
    total_deleted += deleted
    total_skipped += skipped
    total_errors += errors

    # 打印总结
    print("\n=== 清理完成 ===")
    print(f"总计: 删除={total_deleted}, 跳过={total_skipped}, 错误={total_errors}")

    if total_errors > 0:
        print("\n⚠️  有文件删除失败，请检查文件权限或是否被其他程序占用")
    elif total_deleted > 0:
        print("\n✅ 根目录清理成功！")
    else:
        print("\n📁 根目录已经很干净，无需清理")

    # 显示剩余的重要文件
    print("\n📋 保留的重要文件:")
    for important_file in IMPORTANT_FILES_TO_KEEP:
        file_path = ROOT_DIR / important_file
        if file_path.exists():
            print(f"  ✓ {important_file}")


if __name__ == "__main__":
    main()

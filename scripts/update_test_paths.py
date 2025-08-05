#!/usr/bin/env python3
"""
更新测试文件中的图像路径引用

将测试文件中的图像路径引用更新为使用fixtures目录
"""

import glob
import os
import re
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 测试目录
TESTS_DIR = ROOT_DIR / "tests"

# 测试数据目录
FIXTURES_DIR = TESTS_DIR / "fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
PERSON_IMAGES_DIR = IMAGES_DIR / "person"
HAIRNET_IMAGES_DIR = IMAGES_DIR / "hairnet"


# 图像文件名列表
def get_image_filenames():
    """获取所有测试图像文件名"""
    person_images = [f.name for f in PERSON_IMAGES_DIR.glob("*") if f.is_file()]
    hairnet_images = [f.name for f in HAIRNET_IMAGES_DIR.glob("*") if f.is_file()]
    return person_images + hairnet_images


# 更新测试文件中的图像路径
def update_test_file_paths(file_path):
    """更新单个测试文件中的图像路径引用"""
    # 读取文件内容
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 原始内容
    original_content = content

    # 获取所有图像文件名
    image_filenames = get_image_filenames()

    # 替换直接引用图像文件的路径
    for image_name in image_filenames:
        # 确定图像类型
        if "hairnet" in image_name.lower():
            fixture_path = f"tests/fixtures/images/hairnet/{image_name}"
        else:
            fixture_path = f"tests/fixtures/images/person/{image_name}"

        # 替换模式1: 直接引用文件名
        pattern1 = f'"{image_name}"'
        if pattern1 in content and not "fixtures" in content:
            content = content.replace(pattern1, f'"{fixture_path}"')

        # 替换模式2: 使用os.path.join
        pattern2 = f"os.path.join\(.*?{image_name}\)"
        if re.search(pattern2, content) and not "fixtures" in content:
            content = re.sub(
                pattern2,
                f'os.path.join(os.path.dirname(__file__), "../{fixture_path}")',
                content,
            )

    # 添加导入Path（如果需要）
    if "from pathlib import Path" not in content and "import pathlib" not in content:
        imports = re.search(r"import.*?\n\n", content, re.DOTALL)
        if imports:
            content = content.replace(
                imports.group(0), imports.group(0) + "from pathlib import Path\n"
            )

    # 添加获取fixtures目录的辅助函数（如果需要）
    if "def get_fixtures_dir" not in content and "fixtures" in content:
        # 查找合适的位置插入函数
        imports_end = re.search(r"import.*?\n\n", content, re.DOTALL)
        if imports_end:
            helper_func = (
                "def get_fixtures_dir():\n"
                '    """获取测试数据目录"""\n'
                '    return Path(__file__).parent.parent / "fixtures"\n\n'
            )
            content = content.replace(
                imports_end.group(0), imports_end.group(0) + helper_func
            )

    # 如果内容有变化，写回文件
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True

    return False


# 更新所有测试文件
def update_all_test_files():
    """更新所有测试文件中的图像路径引用"""
    # 获取所有测试文件
    test_files = []
    for pattern in ["test_*.py", "*_test.py"]:
        test_files.extend(TESTS_DIR.glob(f"**/{pattern}"))

    # 更新计数
    updated_count = 0

    for test_file in test_files:
        if update_test_file_paths(test_file):
            print(f"已更新: {test_file}")
            updated_count += 1
        else:
            print(f"无需更新: {test_file}")

    return updated_count


# 主函数
def main():
    """主函数"""
    print("=== 更新测试文件中的图像路径引用 ===")

    # 更新所有测试文件
    updated_count = update_all_test_files()
    print(f"\n已更新 {updated_count} 个测试文件")

    print("\n测试文件路径更新完成！")


if __name__ == "__main__":
    main()

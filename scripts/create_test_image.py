#!/usr/bin/env python3
"""
创建测试图像脚本

用于生成洗手检测演示所需的测试图像
"""

from pathlib import Path

import cv2
import numpy as np


def create_test_image() -> None:
    """创建一个简单的测试图像"""
    # 创建一个640x480的彩色图像
    height, width = 480, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 设置背景色为浅灰色
    image[:] = (200, 200, 200)

    # 绘制一个简单的人形轮廓
    # 头部 (圆形)
    cv2.circle(image, (320, 120), 50, (150, 150, 150), -1)

    # 身体 (矩形)
    cv2.rectangle(image, (270, 170), (370, 350), (150, 150, 150), -1)

    # 左臂
    cv2.rectangle(image, (220, 180), (270, 280), (150, 150, 150), -1)

    # 右臂
    cv2.rectangle(image, (370, 180), (420, 280), (150, 150, 150), -1)

    # 左手 (圆形)
    cv2.circle(image, (245, 300), 25, (120, 120, 120), -1)

    # 右手 (圆形)
    cv2.circle(image, (395, 300), 25, (120, 120, 120), -1)

    # 左腿
    cv2.rectangle(image, (290, 350), (320, 450), (150, 150, 150), -1)

    # 右腿
    cv2.rectangle(image, (350, 350), (380, 450), (150, 150, 150), -1)

    # 添加一些文字
    cv2.putText(
        image, "Test Person", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
    )

    # 确保目录存在
    output_dir = Path("tests/fixtures/images/person")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存图像
    output_path = output_dir / "test_person.jpg"
    cv2.imwrite(str(output_path), image)

    print(f"测试图像已创建: {output_path}")
    print(f"图像尺寸: {width}x{height}")


if __name__ == "__main__":
    create_test_image()

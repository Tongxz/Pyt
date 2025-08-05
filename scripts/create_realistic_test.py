#!/usr/bin/env python3
"""
创建更真实的测试图像用于ROI可视化
"""

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


def create_realistic_test_image():
    """创建一个更真实的测试图像"""
    # 创建一个更大的图像
    width, height = 800, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 240  # 浅色背景

    # 添加地板纹理
    for i in range(0, height, 20):
        cv2.line(image, (0, i), (width, i), (220, 220, 220), 1)
    for i in range(0, width, 20):
        cv2.line(image, (i, 0), (i, height), (220, 220, 220), 1)

    # 人体1 - 左侧，佩戴浅蓝色发网
    # 身体
    cv2.rectangle(image, (150, 250), (250, 550), (100, 120, 140), -1)
    # 头部
    cv2.ellipse(image, (200, 200), (40, 50), 0, 0, 360, (220, 200, 180), -1)
    # 浅蓝色发网
    cv2.ellipse(image, (200, 180), (35, 25), 0, 0, 360, (180, 220, 255), -1)
    # 添加网状纹理
    for i in range(170, 230, 8):
        for j in range(160, 200, 8):
            if (i + j) % 16 == 0:
                cv2.circle(image, (i, j), 1, (120, 180, 220), -1)

    # 人体2 - 中间，佩戴白色发网
    # 身体
    cv2.rectangle(image, (350, 240), (450, 560), (90, 110, 130), -1)
    # 头部
    cv2.ellipse(image, (400, 190), (42, 52), 0, 0, 360, (210, 190, 170), -1)
    # 白色发网
    cv2.ellipse(image, (400, 170), (37, 27), 0, 0, 360, (250, 250, 250), -1)
    # 添加网状纹理
    for i in range(370, 430, 6):
        for j in range(150, 190, 6):
            if (i + j) % 12 == 0:
                cv2.circle(image, (i, j), 1, (200, 200, 200), -1)

    # 人体3 - 右侧，无发网
    # 身体
    cv2.rectangle(image, (550, 260), (650, 570), (110, 130, 150), -1)
    # 头部（无发网）
    cv2.ellipse(image, (600, 210), (38, 48), 0, 0, 360, (200, 180, 160), -1)
    # 头发
    cv2.ellipse(image, (600, 190), (35, 30), 0, 0, 360, (80, 60, 40), -1)

    # 添加一些环境细节
    # 设备或背景物体
    cv2.rectangle(image, (50, 100), (150, 200), (180, 180, 180), -1)
    cv2.rectangle(image, (650, 80), (750, 180), (170, 170, 170), -1)

    # 添加轻微的噪声
    noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def test_with_realistic_image():
    """使用更真实的图像测试ROI可视化"""
    print("创建更真实的测试图像...")
    realistic_image = create_realistic_test_image()

    # 保存图像
    test_image_path = "realistic_test_image.jpg"
    cv2.imwrite(test_image_path, cv2.cvtColor(realistic_image, cv2.COLOR_RGB2BGR))
    print(f"真实测试图像已保存: {test_image_path}")

    # 显示图像预览
    plt.figure(figsize=(12, 8))
    plt.imshow(realistic_image)
    plt.title("Realistic Test Image with Simulated Hairnets")
    plt.axis("off")

    # 标注人体位置
    ax = plt.gca()
    # 人体1
    rect1 = Rectangle(
        (150, 150), 100, 400, linewidth=2, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect1)
    plt.text(
        200,
        140,
        "Person 1\n(Blue Hairnet)",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )

    # 人体2
    rect2 = Rectangle(
        (350, 140), 100, 420, linewidth=2, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect2)
    plt.text(
        400,
        130,
        "Person 2\n(White Hairnet)",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.7),
    )

    # 人体3
    rect3 = Rectangle(
        (550, 160), 100, 410, linewidth=2, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect3)
    plt.text(
        600,
        150,
        "Person 3\n(No Hairnet)",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig("realistic_test_preview.png", dpi=150, bbox_inches="tight")
    plt.show()

    return test_image_path


if __name__ == "__main__":
    print("=== 创建真实测试图像 ===")
    image_path = test_with_realistic_image()
    print(f"\n测试图像已创建: {image_path}")
    print("\n现在可以使用以下命令测试ROI可视化:")
    print(
        f"python visualize_roi.py --image {image_path} --save realistic_roi_analysis.png"
    )

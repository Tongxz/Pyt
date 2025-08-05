#!/usr/bin/env python3
"""
创建带有发网特征的测试图片
"""

import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw


def create_image_with_hairnet():
    """创建一个包含发网特征的测试图片"""
    # 创建一个200x200的白色背景图像
    img = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(img)

    # 绘制人体轮廓 (简化的人形)
    # 头部 (圆形)
    draw.ellipse([70, 20, 130, 80], fill=(220, 180, 160))  # 肤色头部

    # 身体 (矩形)
    draw.rectangle([85, 80, 115, 150], fill=(100, 100, 200))  # 蓝色衣服

    # 在头部绘制发网图案
    # 绘制网格状结构模拟发网
    for x in range(75, 125, 8):  # 垂直线
        draw.line([(x, 25), (x, 75)], fill=(50, 50, 50), width=1)

    for y in range(30, 70, 8):  # 水平线
        draw.line([(75, y), (125, y)], fill=(50, 50, 50), width=1)

    # 添加一些不规则的网状结构
    for i in range(10):
        x1 = np.random.randint(75, 125)
        y1 = np.random.randint(25, 75)
        x2 = x1 + np.random.randint(-5, 5)
        y2 = y1 + np.random.randint(-5, 5)
        draw.line([(x1, y1), (x2, y2)], fill=(40, 40, 40), width=1)

    return img


def create_image_without_hairnet():
    """创建一个没有发网的测试图片"""
    img = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(img)

    # 绘制人体轮廓
    # 头部 (圆形) - 没有发网
    draw.ellipse([70, 20, 130, 80], fill=(220, 180, 160))  # 肤色头部

    # 头发 (深色区域)
    draw.ellipse([75, 25, 125, 55], fill=(50, 30, 20))  # 深棕色头发

    # 身体
    draw.rectangle([85, 80, 115, 150], fill=(200, 100, 100))  # 红色衣服

    return img


def test_both_cases():
    """测试有发网和无发网两种情况"""
    url = "http://localhost:8000/api/v1/detect/hairnet"

    print("=== 测试1: 带发网的图片 ===")
    # 测试带发网的图片
    img_with_hairnet = create_image_with_hairnet()
    img_bytes = io.BytesIO()
    img_with_hairnet.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    files = {"file": ("with_hairnet.jpg", img_bytes, "image/jpeg")}

    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            result = response.json()
            print(f"API原始响应: {result}")

            detections = result.get("detections", {})
            total_persons = detections.get("total_persons", 0)
            hairnet_count = detections.get("persons_with_hairnet", 0)
            compliance_rate = detections.get("compliance_rate", 0.0)

            print(f"检测到 {total_persons} 个人")
            print(f"佩戴发网: {hairnet_count} 人")
            print(f"合规率: {compliance_rate:.2f}%")

            for i, detection in enumerate(detections.get("detections", [])):
                print(
                    f"  人员{i+1}: 发网={detection['has_hairnet']}, 置信度={detection['confidence']:.3f}"
                )

                # 显示调试信息
                if "debug_info" in detection:
                    debug = detection["debug_info"]
                    print(f"    边缘密度: {debug.get('edge_density', 0):.4f}")
                    print(f"    轮廓数量: {debug.get('contour_count', 0)}")
                    print(f"    浅蓝色比例: {debug.get('light_blue_ratio', 0):.4f}")
                    print(f"    浅色比例: {debug.get('light_color_ratio', 0):.4f}")
                    print(f"    上部边缘密度: {debug.get('upper_edge_density', 0):.4f}")
                    print(f"    综合得分: {debug.get('total_score', 0):.4f}")
                    print(
                        f"    检测条件: 浅蓝={debug.get('has_light_blue_hairnet', False)}, 一般={debug.get('has_general_hairnet', False)}, 浅色={debug.get('has_light_hairnet', False)}, 基础={debug.get('has_basic_hairnet', False)}"
                    )
                    if "error" in debug:
                        print(f"    错误信息: {debug['error']}")
                else:
                    print(f"    边缘密度: {detection.get('edge_density', 0):.4f}")
                    print(f"    轮廓数量: {detection.get('contour_count', 0)}")
        else:
            print(f"请求失败: {response.status_code}")
    except Exception as e:
        print(f"测试失败: {e}")

    print("\n=== 测试2: 不带发网的图片 ===")
    # 测试不带发网的图片
    img_without_hairnet = create_image_without_hairnet()
    img_bytes = io.BytesIO()
    img_without_hairnet.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    files = {"file": ("without_hairnet.jpg", img_bytes, "image/jpeg")}

    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            result = response.json()
            detections = result.get("detections", {})
            print(f"检测到 {detections.get('total_persons', 0)} 个人")
            print(f"佩戴发网: {detections.get('persons_with_hairnet', 0)} 人")
            print(f"合规率: {detections.get('compliance_rate', 0):.2%}")

            # 显示详细检测结果
            for i, detection in enumerate(detections.get("detections", [])):
                print(
                    f"  人员{i+1}: 发网={detection['has_hairnet']}, 置信度={detection['confidence']:.3f}"
                )
        else:
            print(f"请求失败: {response.status_code}")
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    print("开始测试发网检测准确性...")
    test_both_cases()
    print("\n测试完成")

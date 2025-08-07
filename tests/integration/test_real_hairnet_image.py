#!/usr/bin/env python3
"""
真实发网图片测试脚本
使用用户提供的真实发网图片进行测试
"""

import base64
import io
from pathlib import Path

import requests
from PIL import Image


def test_real_hairnet_image():
    """测试真实发网图片"""
    # 使用tests目录下的测试图片
    test_image_path = "tests/fixtures/images/person/test_person.png"

    print("=== 真实发网图片测试 ===")
    print(f"使用测试图片: {test_image_path}")

    try:
        # 尝试读取测试图片文件
        with open(test_image_path, "rb") as f:
            image_data = f.read()

        # 发送到API进行检测
        url = "http://localhost:8000/api/v1/detect/hairnet"
        files = {"file": ("test_person.png", image_data, "image/png")}

        response = requests.post(url, files=files)

        if response.status_code == 200:
            result = response.json()
            print("API原始响应:", result)

            detections = result.get("detections", {})
            print(f"检测到 {detections.get('total_persons', 0)} 个人")
            print(f"佩戴发网: {detections.get('persons_with_hairnet', 0)} 人")
            print(f"合规率: {detections.get('compliance_rate', 0):.2%}")
            print(f"平均置信度: {detections.get('average_confidence', 0):.3f}")

            # 显示详细检测结果
            for i, detection in enumerate(detections.get("detections", [])):
                print(
                    f"  人员{i+1}: 发网={detection['has_hairnet']}, 置信度={detection['confidence']:.3f}"
                )

                # 如果有调试信息，显示出来
                if "debug_info" in detection:
                    debug = detection["debug_info"]
                    print(f"    边缘密度: {debug.get('edge_density', 0):.4f}")
                    print(f"    轮廓数量: {debug.get('contour_count', 0)}")
                    print(f"    浅蓝色比例: {debug.get('light_blue_ratio', 0):.4f}")
                    print(f"    浅色比例: {debug.get('light_color_ratio', 0):.4f}")
                    print(f"    上部边缘密度: {debug.get('upper_edge_density', 0):.4f}")
                    print(f"    综合得分: {debug.get('total_score', 0):.4f}")

                    # 显示各种发网判断条件
                    conditions = [
                        ("浅蓝色发网", debug.get("has_light_blue_hairnet", False)),
                        ("一般发网", debug.get("has_general_hairnet", False)),
                        ("浅色发网", debug.get("has_light_hairnet", False)),
                        ("基础发网", debug.get("has_basic_hairnet", False)),
                    ]

                    for condition_name, condition_value in conditions:
                        print(f"    {condition_name}: {condition_value}")
        else:
            print(f"测试失败: {response.text}")

    except FileNotFoundError:
        print(f"错误: 未找到测试图片文件 '{test_image_path}'")
        print("请确保tests/fixtures/images目录下有测试图片")
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    test_real_hairnet_image()

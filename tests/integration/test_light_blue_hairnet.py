#!/usr/bin/env python3
"""
浅蓝色发网检测测试脚本
专门测试浅蓝色发网的检测效果
"""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw


def create_light_blue_hairnet_image():
    """
    创建一个更真实的浅蓝色发网图像
    """
    # 创建一个512x512的图像
    img = Image.new("RGB", (512, 512), color=(240, 240, 240))  # 浅灰色背景
    draw = ImageDraw.Draw(img)

    # 绘制人脸轮廓（椭圆形）
    face_center = (256, 300)
    face_width, face_height = 120, 150
    face_bbox = [
        face_center[0] - face_width // 2,
        face_center[1] - face_height // 2,
        face_center[0] + face_width // 2,
        face_center[1] + face_height // 2,
    ]
    draw.ellipse(face_bbox, fill=(255, 220, 177))  # 肤色

    # 绘制头发区域
    hair_bbox = [
        face_center[0] - face_width // 2 - 20,
        face_center[1] - face_height // 2 - 80,
        face_center[0] + face_width // 2 + 20,
        face_center[1] - face_height // 2 + 40,
    ]
    draw.ellipse(hair_bbox, fill=(101, 67, 33))  # 棕色头发

    # 绘制浅蓝色发网 - 网格状结构
    light_blue = (173, 216, 230)  # 浅蓝色

    # 发网覆盖区域
    net_left = face_center[0] - face_width // 2 - 15
    net_right = face_center[0] + face_width // 2 + 15
    net_top = face_center[1] - face_height // 2 - 75
    net_bottom = face_center[1] - face_height // 2 + 35

    # 绘制水平网格线
    for y in range(net_top, net_bottom, 8):
        draw.line([(net_left, y), (net_right, y)], fill=light_blue, width=2)

    # 绘制垂直网格线
    for x in range(net_left, net_right, 8):
        draw.line([(x, net_top), (x, net_bottom)], fill=light_blue, width=2)

    # 添加一些不规则的网格线以模拟真实发网
    for i in range(10):
        x1 = np.random.randint(net_left, net_right)
        y1 = np.random.randint(net_top, net_bottom)
        x2 = x1 + np.random.randint(-15, 15)
        y2 = y1 + np.random.randint(-15, 15)
        draw.line([(x1, y1), (x2, y2)], fill=light_blue, width=1)

    # 转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_cv


def test_light_blue_detection():
    """
    测试浅蓝色发网检测
    """
    print("创建浅蓝色发网测试图像...")

    # 创建测试图像
    test_image = create_light_blue_hairnet_image()

    # 保存图像用于调试
    cv2.imwrite("/Users/zhou/Code/python/Pyt/test_light_blue_hairnet.jpg", test_image)
    print("测试图像已保存为: test_light_blue_hairnet.jpg")

    # 编码图像
    _, buffer = cv2.imencode(".jpg", test_image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    try:
        # 将图像编码为字节流
        _, buffer = cv2.imencode(".jpg", test_image)
        img_bytes = buffer.tobytes()

        # 发送检测请求（使用文件上传格式）
        files = {"file": ("test_image.jpg", img_bytes, "image/jpeg")}
        response = requests.post(
            "http://localhost:8000/api/v1/detect/hairnet", files=files, timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n=== 浅蓝色发网检测结果 ===")

            if "detections" in result:
                detections = result["detections"]
                total_persons = detections.get("detection_count", 0)
                hairnet_count = detections.get("hairnet_count", 0)
                compliance_rate = detections.get("compliance_rate", 0.0)
                avg_confidence = detections.get("average_confidence", 0.0)

                print(f"检测到 {total_persons} 个人")
                print(f"佩戴发网: {hairnet_count} 人")
                print(f"合规率: {compliance_rate:.2f}%")
                print(f"平均置信度: {avg_confidence:.3f}")

                # 详细检测信息
                if "persons" in detections:
                    for i, person in enumerate(detections["persons"]):
                        has_hairnet = person.get("has_hairnet", False)
                        confidence = person.get("confidence", 0.0)
                        print(f"  人员 {i+1}: 发网={has_hairnet}, 置信度={confidence:.3f}")

                        # 如果有调试信息，显示特征分析
                        if "debug_info" in person:
                            debug = person["debug_info"]
                            print(f"    边缘密度: {debug.get('edge_density', 0):.4f}")
                            print(f"    轮廓数量: {debug.get('contour_count', 0)}")
                            print(f"    浅蓝色比例: {debug.get('light_blue_ratio', 0):.4f}")
                            print(f"    浅色比例: {debug.get('light_color_ratio', 0):.4f}")

                if hairnet_count > 0:
                    print("\n✅ 成功检测到浅蓝色发网！")
                else:
                    print("\n❌ 未能检测到浅蓝色发网")
            else:
                print("API响应格式异常")
                print(f"响应内容: {result}")
        else:
            print(f"API请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")

    except Exception as e:
        print(f"检测过程中发生错误: {e}")


if __name__ == "__main__":
    print("开始浅蓝色发网检测测试...")
    test_light_blue_detection()
    print("\n测试完成")

#!/usr/bin/env python3
"""
使用包含人体的测试图像来验证综合检测API
"""

import json
from pathlib import Path

import cv2
import numpy as np
import requests


def create_test_image_with_person():
    """
    创建一个包含人形轮廓的测试图像
    """
    # 创建640x480的图像
    image = np.ones((480, 640, 3), dtype=np.uint8) * 240  # 浅灰色背景

    # 绘制一个简单的人形轮廓
    # 头部 (圆形)
    cv2.circle(image, (320, 100), 40, (100, 100, 100), -1)

    # 身体 (矩形)
    cv2.rectangle(image, (280, 140), (360, 300), (100, 100, 100), -1)

    # 手臂
    cv2.rectangle(image, (240, 160), (280, 220), (100, 100, 100), -1)  # 左臂
    cv2.rectangle(image, (360, 160), (400, 220), (100, 100, 100), -1)  # 右臂

    # 腿部
    cv2.rectangle(image, (290, 300), (320, 400), (100, 100, 100), -1)  # 左腿
    cv2.rectangle(image, (320, 300), (350, 400), (100, 100, 100), -1)  # 右腿

    # 添加一些噪声使图像更真实
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)

    return image


def test_comprehensive_detection_with_person():
    """
    使用包含人体的图像测试综合检测API
    """
    api_url = "http://localhost:8000/api/v1/detect/comprehensive"

    # 创建测试图像
    test_image = create_test_image_with_person()

    # 保存测试图像以便查看
    cv2.imwrite("/Users/zhou/Code/python/Pyt/test_person_image.jpg", test_image)
    print("📸 测试图像已保存为 test_person_image.jpg")

    # 编码图像
    _, buffer = cv2.imencode(".jpg", test_image)

    # 准备文件数据
    files = {"file": ("test_person.jpg", buffer.tobytes(), "image/jpeg")}

    try:
        print("\n🔄 正在调用综合检测API...")
        response = requests.post(api_url, files=files, timeout=30)

        print(f"📊 响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ API调用成功!")

            # 显示关键统计信息
            print(f"\n📊 检测统计:")
            print(f"   👥 总人数: {result.get('total_persons', 0)}")

            statistics = result.get("statistics", {})
            print(f"   🧢 佩戴发网人数: {statistics.get('persons_with_hairnet', 0)}")
            print(f"   🧼 洗手人数: {statistics.get('persons_handwashing', 0)}")
            print(f"   🧴 消毒人数: {statistics.get('persons_sanitizing', 0)}")

            # 显示检测详情
            detections = result.get("detections", [])
            print(f"\n🔍 检测详情 ({len(detections)} 个目标):")
            for i, detection in enumerate(detections):
                print(
                    f"   {i+1}. 类别: {detection.get('class', 'unknown')}, "
                    f"置信度: {detection.get('confidence', 0):.3f}, "
                    f"边界框: {detection.get('bbox', [])}"
                )

            # 显示处理时间
            processing_times = result.get("processing_times", {})
            if processing_times:
                print(f"\n⏱️ 处理时间:")
                for key, value in processing_times.items():
                    print(f"   {key}: {value:.3f}s")

            # 检查是否有标注图像
            if result.get("annotated_image"):
                print("\n🖼️ 包含标注图像 (base64编码)")

                # 可选：解码并保存标注图像
                try:
                    import base64

                    img_data = base64.b64decode(result["annotated_image"])
                    with open(
                        "/Users/zhou/Code/python/Pyt/test_annotated_result.jpg", "wb"
                    ) as f:
                        f.write(img_data)
                    print("📸 标注结果已保存为 test_annotated_result.jpg")
                except Exception as e:
                    print(f"❌ 保存标注图像失败: {e}")

            return True

        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("🧪 使用包含人体的图像测试综合检测API...")

    if test_comprehensive_detection_with_person():
        print("\n🎉 测试完成! 请检查前端页面是否能正确显示这些数据。")
        print("\n💡 建议:")
        print("   1. 打开 http://localhost:8000/frontend/index.html")
        print("   2. 上传 test_person_image.jpg 文件")
        print("   3. 点击'开始检测'按钮")
        print("   4. 检查显示的统计数据和标注图像是否正确")
    else:
        print("\n💥 测试失败!")
        exit(1)

#!/usr/bin/env python3
"""
测试综合检测API的调用和返回数据
"""

import base64
import json
from pathlib import Path

import requests


def test_comprehensive_detection_api():
    """
    测试综合检测API
    """
    api_url = "http://localhost:8000/api/v1/detect/comprehensive"

    # 创建一个简单的测试图像（1x1像素的白色图像）
    import cv2
    import numpy as np

    # 创建测试图像
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, buffer = cv2.imencode(".jpg", test_image)

    # 准备文件数据
    files = {"file": ("test_image.jpg", buffer.tobytes(), "image/jpeg")}

    try:
        print("🔄 正在调用综合检测API...")
        response = requests.post(api_url, files=files, timeout=30)

        print(f"📊 响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ API调用成功!")
            print("\n📋 返回数据结构:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # 检查必要的字段
            required_fields = ["total_persons", "statistics", "detections"]
            missing_fields = []

            for field in required_fields:
                if field not in result:
                    missing_fields.append(field)

            if missing_fields:
                print(f"\n❌ 缺少必要字段: {missing_fields}")
                return False

            # 检查statistics字段的结构
            statistics = result.get("statistics", {})
            stats_fields = [
                "persons_with_hairnet",
                "persons_handwashing",
                "persons_sanitizing",
            ]
            missing_stats = []

            for field in stats_fields:
                if field not in statistics:
                    missing_stats.append(field)

            if missing_stats:
                print(f"\n❌ statistics字段缺少: {missing_stats}")
                return False

            # 检查annotated_image字段
            if "annotated_image" in result and result["annotated_image"]:
                try:
                    # 验证base64编码
                    base64.b64decode(result["annotated_image"])
                    print("✅ annotated_image字段格式正确 (base64)")
                except Exception as e:
                    print(f"❌ annotated_image字段格式错误: {e}")
                    return False

            print("\n✅ 所有必要字段都存在且格式正确!")
            return True

        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ 网络请求失败: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        if "response" in locals():
            print(f"响应内容: {response.text}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_api_health():
    """
    测试API健康状态
    """
    health_url = "http://localhost:8000/health"

    try:
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            print("✅ API服务健康状态正常")
            return True
        else:
            print(f"❌ API服务健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到API服务: {e}")
        return False


if __name__ == "__main__":
    print("🧪 开始测试综合检测API...\n")

    # 首先检查API健康状态
    if not test_api_health():
        print("\n❌ API服务不可用，请确保服务器正在运行")
        exit(1)

    print()

    # 测试综合检测API
    if test_comprehensive_detection_api():
        print("\n🎉 综合检测API测试通过!")
    else:
        print("\n💥 综合检测API测试失败!")
        exit(1)

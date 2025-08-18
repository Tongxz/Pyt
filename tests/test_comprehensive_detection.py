#!/usr/bin/env python3
"""
综合检测功能测试脚本
测试后端API的综合检测功能是否正常工作
"""

import json
import os
from pathlib import Path

import requests

# 禁用代理设置，确保本地请求不通过代理
proxies = {"http": "", "https": ""}


def test_comprehensive_detection():
    """测试综合检测API"""

    # API端点
    url = "http://localhost:8000/api/v1/detect/comprehensive"

    # 测试图像文件路径
    test_images = [
        "fixtures/images/person/test_person.png",
        "fixtures/images/hairnet/7月23日.png",
    ]

    print("=== 综合检测功能测试 ===")
    print(f"API端点: {url}")
    print()

    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"❌ 测试图像文件不存在: {image_path}")
            continue

        print(f"📸 测试图像: {image_path}")

        try:
            # 准备文件上传
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "image/png")}
                data = {"record_process": "true"}

                # 发送POST请求
                response = requests.post(
                    url, files=files, data=data, timeout=30, proxies=proxies
                )

                print(f"   状态码: {response.status_code}")

                if response.status_code == 200:
                    try:
                        result = response.json()
                        print(f"   ✅ 检测成功")
                        print(
                            f"   响应数据: {json.dumps(result, indent=2, ensure_ascii=False)}"
                        )
                    except json.JSONDecodeError:
                        print(f"   ⚠️  响应不是有效的JSON格式")
                        print(f"   响应内容: {response.text[:200]}...")
                else:
                    print(f"   ❌ 检测失败")
                    print(f"   错误信息: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"   ❌ 请求异常: {e}")
        except Exception as e:
            print(f"   ❌ 未知错误: {e}")

        print("-" * 50)


def test_health_check():
    """测试健康检查端点"""

    print("=== 健康检查测试 ===")

    try:
        response = requests.get(
            "http://localhost:8000/health", timeout=5, proxies=proxies
        )
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            try:
                result = response.json()
                print(f"✅ 健康检查成功: {result}")
            except json.JSONDecodeError:
                print(f"⚠️  响应不是JSON格式: {response.text}")
        else:
            print(f"❌ 健康检查失败: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ 健康检查请求异常: {e}")

    print()


if __name__ == "__main__":
    # 首先测试健康检查
    test_health_check()

    # 然后测试综合检测功能
    test_comprehensive_detection()

    print("\n=== 测试完成 ===")

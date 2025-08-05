#!/usr/bin/env python3

import json
from pathlib import Path

import requests


def test_hairnet_detection_api():
    """测试发网检测API"""
    url = "http://localhost:8000/api/v1/detect/hairnet"

    # 测试图片路径
    image_path = "test_person.jpg"

    try:
        # 发送POST请求
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)

        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")

        if response.status_code == 200:
            result = response.json()
            print("\n=== API响应内容 ===")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"请求失败: {response.text}")

    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    test_hairnet_detection_api()

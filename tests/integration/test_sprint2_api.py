#!/usr/bin/env python3
"""
Sprint 2 API 完整测试脚本
测试发网检测功能的所有API接口
"""

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict

import requests


def test_health_check() -> bool:
    """测试健康检查接口"""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"健康检查 - 状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"健康检查 - 响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result.get("detector_ready", False)
        return False
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False


def encode_image(image_path: str) -> str:
    """将图片编码为base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_hairnet_detection(image_path: str) -> Dict[str, Any]:
    """测试发网检测API"""
    try:
        # 使用文件上传方式
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(
                "http://localhost:8000/api/v1/detect/hairnet", files=files
            )

        print(f"\n=== 发网检测测试 ({os.path.basename(image_path)}) ===")
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"检测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result
        else:
            print(f"错误: {response.text}")
            return {}
    except Exception as e:
        print(f"发网检测测试失败: {e}")
        return {}


def test_detection_history() -> Dict[str, Any]:
    """测试检测历史查询API"""
    try:
        response = requests.get("http://localhost:8000/api/statistics/history")
        print(f"\n=== 检测历史查询测试 ===")
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            # 直接打印结果，不假设特定的数据结构
            print(f"历史记录: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result
        else:
            print(f"错误: {response.text}")
            return {}
    except Exception as e:
        print(f"检测历史查询失败: {e}")
        return {}


def main():
    """主测试函数"""
    print("=" * 60)
    print("Sprint 2 发网检测功能完整测试")
    print("=" * 60)

    # 1. 健康检查
    if not test_health_check():
        print("❌ 服务未就绪，测试终止")
        return

    # 2. 查找测试图片
    test_images = [
        "tests/fixtures/images/person/test_person.png",
        "tests/fixtures/images/hairnet/7月23日.png",
        "tests/fixtures/images/person/test_person.png",
        "tests/fixtures/images/hairnet/7月23日.png",
        "tests/fixtures/images/person/test_person.png",
    ]

    available_images = []
    for img in test_images:
        if os.path.exists(img):
            available_images.append(img)

    if not available_images:
        print("❌ 未找到测试图片")
        return

    print(f"\n找到 {len(available_images)} 张测试图片: {available_images}")

    # 3. 测试发网检测
    detection_results = []
    for img_path in available_images[:3]:  # 测试前3张图片
        result = test_hairnet_detection(img_path)
        if result:
            detection_results.append(result)

    # 4. 测试检测历史
    history_result = test_detection_history()

    # 5. 测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"✅ 健康检查: 通过")
    print(f"✅ 发网检测: {len(detection_results)}/{len(available_images[:3])} 成功")
    print(f"✅ 历史查询: {'通过' if history_result else '失败'}")

    # 统计检测结果
    if detection_results:
        hairnet_count = sum(1 for r in detection_results if r.get("has_hairnet", False))
        print(f"📊 发网检测统计: {hairnet_count}/{len(detection_results)} 张图片检测到发网")

    print("\n🎉 Sprint 2 功能测试完成!")


if __name__ == "__main__":
    main()

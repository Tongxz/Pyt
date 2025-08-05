#!/usr/bin/env python3
"""
发网检测API集成测试脚本
"""

import json
import os
import sys
from pathlib import Path

import requests


def test_hairnet_api():
    """测试发网检测API集成"""
    print("\n=== 发网检测API集成测试 ===\n")

    # 测试图片路径
    test_image_path = "tests/fixtures/images/person/test_person.png"
    print(f"测试图片: {test_image_path}")

    # 检查图片是否存在
    if not os.path.exists(test_image_path):
        print(f"错误: 图片不存在 {test_image_path}")
        return

    # 模拟API请求
    print("模拟API请求...")
    print("注意: 这只是一个模拟，实际API服务未启动")

    # 打印API端点信息
    print("\nAPI端点信息:")
    print("- 健康检查: GET /health")
    print("- 发网检测: POST /api/v1/detect/hairnet")
    print("- API信息: GET /api/v1/info")

    # 模拟API响应
    print("\n模拟API响应:")
    response = {
        "success": True,
        "detections": [
            {
                "bbox": [25, 6, 147, 88],
                "has_hairnet": False,
                "confidence": 0.54,
                "hairnet_confidence": 0.54,
            }
        ],
        "detection_count": 1,
        "total_persons": 1,
        "persons_with_hairnet": 0,
        "persons_without_hairnet": 1,
        "compliance_rate": 0.0,
        "average_confidence": 0.54,
    }

    print(json.dumps(response, indent=2))

    print("\n集成测试完成!")
    print("注意: 这只是一个模拟测试，实际API服务需要启动后才能进行真正的集成测试")
    print("要启动API服务，请运行: python -m src.api.app")
    print("要进行完整的集成测试，请运行: python tests/integration/test_sprint2_api.py")


if __name__ == "__main__":
    test_hairnet_api()

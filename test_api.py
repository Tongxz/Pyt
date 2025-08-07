#!/usr/bin/env python3
"""
测试API接口功能
"""

import json
import os

import requests


def test_health_check():
    """测试健康检查接口"""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"健康检查状态码: {response.status_code}")
        print(f"健康检查响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False


def test_api_info():
    """测试API信息接口"""
    try:
        response = requests.get("http://localhost:8000/api/info")
        print(f"API信息状态码: {response.status_code}")
        print(f"API信息响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"API信息获取失败: {e}")
        return False


def test_comprehensive_detection():
    """测试综合检测接口"""
    image_path = "tests/fixtures/images/person/test_person.png"

    if not os.path.exists(image_path):
        print(f"测试图片不存在: {image_path}")
        return False

    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {"record_process": "false"}
            response = requests.post(
                "http://localhost:8000/api/v1/detect/comprehensive",
                files=files,
                data=data,
            )

        print(f"综合检测状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"检测到人数: {result.get('total_persons', 0)}")

            stats = result.get("statistics", {})
            print(
                f"发网统计: 佩戴={stats.get('persons_with_hairnet', 0)}, 未佩戴={stats.get('persons_without_hairnet', 0)}"
            )
            print(
                f"行为统计: 洗手={stats.get('persons_handwashing', 0)}, 消毒={stats.get('persons_sanitizing', 0)}"
            )
            print(
                f"合规率: 发网={stats.get('hairnet_compliance_rate', 0):.2%}, 洗手={stats.get('handwash_rate', 0):.2%}, 消毒={stats.get('sanitize_rate', 0):.2%}"
            )

            processing_time = result.get("processing_time", {})
            print(f"处理时间: 总计={processing_time.get('total_time', 0):.3f}秒")

            # 检查是否有标注图像
            has_image = (
                "annotated_image" in result and result["annotated_image"] is not None
            )
            print(f"包含标注图像: {has_image}")

            return True
        else:
            print(f"检测失败: {response.text}")
            return False

    except Exception as e:
        print(f"综合检测失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=== API接口测试 ===")

    # 测试健康检查
    print("\n1. 测试健康检查接口")
    health_ok = test_health_check()

    # 测试API信息
    print("\n2. 测试API信息接口")
    info_ok = test_api_info()

    # 测试综合检测
    print("\n3. 测试综合检测接口")
    detection_ok = test_comprehensive_detection()

    # 总结
    print("\n=== 测试结果总结 ===")
    print(f"健康检查: {'✓' if health_ok else '✗'}")
    print(f"API信息: {'✓' if info_ok else '✗'}")
    print(f"综合检测: {'✓' if detection_ok else '✗'}")

    all_passed = health_ok and info_ok and detection_ok
    print(f"\n整体状态: {'所有接口正常' if all_passed else '部分接口异常'}")

    return all_passed


if __name__ == "__main__":
    main()

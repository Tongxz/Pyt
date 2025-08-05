#!/usr/bin/env python3
"""
详细发网检测分析脚本
用于深入分析发网检测算法的性能和准确性
"""

import json
import os

import requests


def get_fixtures_dir():
    """获取测试数据目录"""
    return Path(__file__).parent.parent / "fixtures"


from pathlib import Path


def test_detailed_hairnet_analysis():
    """详细分析发网检测结果"""
    print("=== 详细发网检测分析 ===")

    # 检查图片文件是否存在
    image_path = "tests/fixtures/images/person/test_person.png"
    if not os.path.exists(image_path):
        print(f"错误: 未找到 '{image_path}' 文件")
        print(f"请将真实发网图片保存为 '{image_path}' 到项目根目录")
        return

    try:
        # 读取图片文件
        with open(image_path, "rb") as f:
            image_data = f.read()

        # 发送到API进行检测
        url = "http://localhost:8000/api/v1/detect/hairnet"
        files = {"file": (image_path, image_data, "image/png")}

        response = requests.post(url, files=files)

        if response.status_code == 200:
            result = response.json()

            print("\n=== 检测结果概览 ===")
            detections = result.get("detections", {})
            print(f"检测到人数: {detections.get('total_persons', 0)}")
            print(f"佩戴发网: {detections.get('persons_with_hairnet', 0)} 人")
            print(f"未佩戴发网: {detections.get('persons_without_hairnet', 0)} 人")
            print(f"合规率: {detections.get('compliance_rate', 0) * 100:.2f}%")
            print(f"平均置信度: {detections.get('average_confidence', 0):.3f}")

            print("\n=== 详细人员分析 ===")
            for i, detection in enumerate(detections.get("detections", []), 1):
                print(f"\n人员 {i}:")
                print(f"  人体位置: {detection.get('bbox', [])}")
                print(f"  头部区域: {detection.get('head_coords', [])}")
                print(f"  ROI策略: {detection.get('roi_strategy', 'unknown')}")
                print(f"  发网检测: {'是' if detection.get('has_hairnet', False) else '否'}")
                print(f"  置信度: {detection.get('confidence', 0):.3f}")

                # 显示调试信息
                if "debug_info" in detection and detection["debug_info"]:
                    debug = detection["debug_info"]
                    print(f"  调试信息:")
                    print(f"    基础边缘密度: {debug.get('basic_edge_density', 0):.4f}")
                    print(f"    敏感边缘密度: {debug.get('sensitive_edge_density', 0):.4f}")
                    print(f"    最终边缘密度: {detection.get('edge_density', 0):.4f}")
                    print(f"    轮廓数量: {debug.get('contour_count', 0)}")
                    print(f"    浅蓝色比例: {debug.get('light_blue_ratio', 0):.4f}")
                    print(f"    浅色比例: {debug.get('light_color_ratio', 0):.4f}")
                    print(f"    上部边缘密度: {debug.get('upper_edge_density', 0):.4f}")
                    print(f"    综合得分: {debug.get('total_score', 0):.3f}")
                    print(f"    检测条件:")
                    print(f"      浅蓝色发网: {debug.get('has_light_blue_hairnet', False)}")
                    print(f"      一般发网: {debug.get('has_general_hairnet', False)}")
                    print(f"      浅色发网: {debug.get('has_light_hairnet', False)}")
                    print(f"      基础发网: {debug.get('has_basic_hairnet', False)}")
                    print(f"      极宽松发网: {debug.get('has_minimal_hairnet', False)}")

            print("\n=== 算法性能评估 ===")
            total_persons = detections.get("total_persons", 0)
            persons_with_hairnet = detections.get("persons_with_hairnet", 0)
            avg_confidence = detections.get("average_confidence", 0)

            if total_persons > 0:
                print(f"检测覆盖率: {total_persons}/4 = {total_persons/4*100:.1f}%")
                print(
                    f"发网识别率: {persons_with_hairnet}/{total_persons} = {persons_with_hairnet/total_persons*100:.1f}%"
                )
                print(f"平均置信度: {avg_confidence:.3f}")

                if avg_confidence > 0.7:
                    print("✓ 算法置信度较高")
                elif avg_confidence > 0.5:
                    print("⚠ 算法置信度中等")
                else:
                    print("✗ 算法置信度较低")

        else:
            print(f"API请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")

    except FileNotFoundError:
        print(f"错误: 未找到 '{image_path}' 文件")
        print(f"请将真实发网图片保存为 '{image_path}' 到项目根目录")
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    test_detailed_hairnet_analysis()

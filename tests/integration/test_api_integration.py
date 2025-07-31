#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API集成测试：验证发网检测API是否正常工作
"""

import requests
import json
import os
import time

def get_fixtures_dir():
    """获取测试数据目录"""
    return Path(__file__).parent.parent / "fixtures"

from pathlib import Path
def test_api_integration():
    """
    测试发网检测API的集成状态
    """
    print("=== API集成测试 ===")
    
    # API端点
    base_url = "http://localhost:8000"
    detect_url = f"{base_url}/api/v1/detect/hairnet"
    stats_url = f"{base_url}/api/statistics/realtime"
    
    # 检查测试图像
    test_image_path = "/Users/zhou/Code/python/Pyt/tests/fixtures/images/person/test_person.png"
    if not os.path.exists(test_image_path):
        print(f"错误: 未找到测试图像 '{test_image_path}'")
        return False
    
    try:
        # 测试健康检查
        print("\n1. 测试API健康状态...")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ API服务正常运行")
            else:
                print(f"✗ API健康检查失败: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ 无法连接到API服务: {e}")
            print("请确保API服务正在运行 (python -m uvicorn src.api.app:app --reload)")
            return False
        
        # 测试图像检测
        print("\n2. 测试图像检测接口...")
        with open(test_image_path, 'rb') as f:
            files = {'file': (test_image_path, f, 'image/png')}
            response = requests.post(detect_url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ 图像检测成功")
            print(f"响应内容: {result}")
            
            # 显示检测结果
            detections_data = result.get('detections', {})
            if isinstance(detections_data, dict):
                print(f"检测到人数: {detections_data.get('total_persons', 0)}")
                print(f"佩戴发网: {detections_data.get('persons_with_hairnet', 0)} 人")
                print(f"未佩戴发网: {detections_data.get('persons_without_hairnet', 0)} 人")
                print(f"合规率: {detections_data.get('compliance_rate', 0) * 100:.2f}%")
                print(f"平均置信度: {detections_data.get('average_confidence', 0):.3f}")
            else:
                print(f"检测到人数: {len(result.get('detections', []))}")
                print("佩戴发网: 0 人")
                print(f"未佩戴发网: {len(result.get('detections', []))} 人")
                print("合规率: 0.00%")
                print("平均置信度: 0.000")
            
            # 检查是否使用了增强ROI
            detections_data = result.get('detections', {})
            detections = detections_data.get('detections', []) if isinstance(detections_data, dict) else []
            enhanced_roi_count = 0
            for detection in detections:
                if detection.get('enhanced_roi_used', False) or detection.get('roi_strategy', '').startswith('enhanced'):
                    enhanced_roi_count += 1
            
            print(f"使用增强ROI的检测: {enhanced_roi_count}/{len(detections)}")
            
            if enhanced_roi_count > 0:
                print("✓ 增强ROI算法已成功集成到API中")
            else:
                print("⚠ 未检测到增强ROI的使用")
                
        else:
            print(f"✗ 图像检测失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
        
        # 测试统计接口
        print("\n3. 测试统计接口...")
        response = requests.get(stats_url, timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print("✓ 统计接口正常")
            print(f"总检测次数: {stats.get('total_detections', 0)}")
            print(f"总检测人数: {stats.get('total_persons', 0)}")
        else:
            print(f"⚠ 统计接口异常: {response.status_code}")
        
        print("\n=== API集成测试完成 ===")
        return True
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        return False

def check_api_server():
    """
    检查API服务器是否运行
    """
    try:
        response = requests.get("http://localhost:8000/health", timeout=3)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    if not check_api_server():
        print("API服务器未运行，请先启动服务器:")
        print("python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000")
        print("\n或者运行:")
        print("cd src && python -m uvicorn api.app:app --reload")
    else:
        success = test_api_integration()
        if success:
            print("\n✓ 所有测试通过，集成成功！")
        else:
            print("\n✗ 部分测试失败")
#!/usr/bin/env python3
"""
调试API响应的简化测试脚本
"""

import requests
import cv2
import json
from io import BytesIO

def test_api_response():
    """测试API响应并打印完整结果"""
    
    # 测试指定的真实图片
    test_image_path = "tests/fixtures/images/hairnet/7月23日.png"
    
    try:
        # 读取图像
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"无法加载图像: {test_image_path}")
            return
            
        print(f"成功加载图像: {test_image_path}")
        print(f"图像尺寸: {image.shape}")
        
        # 将图像编码为JPEG格式
        _, buffer = cv2.imencode('.jpg', image)
        
        # 准备文件数据
        files = {
            'file': ('test.jpg', BytesIO(buffer.tobytes()), 'image/jpeg')
        }
        
        # 发送请求到综合检测API
        print("\n发送API请求...")
        response = requests.post(
            'http://localhost:8000/api/v1/detect/comprehensive',
            files=files,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("\n=== 完整API响应 ===")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                print("\n=== 关键字段解析 ===")
                print(f"total_persons: {result.get('total_persons', 'NOT_FOUND')}")
                print(f"statistics: {result.get('statistics', 'NOT_FOUND')}")
                print(f"processing_time: {result.get('processing_time', 'NOT_FOUND')}")
                
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                print(f"原始响应内容: {response.text[:500]}...")
        else:
            print(f"API请求失败: {response.text}")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_response()
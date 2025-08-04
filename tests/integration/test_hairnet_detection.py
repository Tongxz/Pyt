#!/usr/bin/env python3
"""
发网检测API测试脚本
"""

import requests
import numpy as np
from PIL import Image
import io
import json

from pathlib import Path
def create_test_image():
    """创建一个简单的测试图片"""
    # 创建一个简单的RGB图像 (200x200像素)
    img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # 添加一些模拟的人脸特征
    # 头部区域 (上半部分)
    img_array[20:80, 60:140] = [220, 180, 160]  # 肤色
    
    # 添加一些边缘特征模拟发网
    for i in range(20, 60, 5):
        for j in range(60, 140, 5):
            img_array[i:i+1, j:j+3] = [100, 100, 100]  # 深色线条
    
    return Image.fromarray(img_array)

def test_hairnet_detection():
    """测试发网检测API"""
    # 创建测试图片
    test_img = create_test_image()
    
    # 将图片转换为字节流
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # 发送请求到API
    url = 'http://localhost:8000/api/v1/detect/hairnet'
    files = {'file': ('test_image.jpg', img_bytes, 'image/jpeg')}
    
    try:
        response = requests.post(url, files=files)
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {response.headers}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n检测结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 分析结果
            detections = result.get('detections', {})
            total_persons = detections.get('total_persons', 0)
            persons_with_hairnet = detections.get('persons_with_hairnet', 0)
            compliance_rate = detections.get('compliance_rate', 0)
            
            print(f"\n检测摘要:")
            print(f"总人数: {total_persons}")
            print(f"佩戴发网人数: {persons_with_hairnet}")
            print(f"合规率: {compliance_rate:.2%}")
            
        else:
            print(f"请求失败: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("无法连接到API服务器，请确保服务器正在运行")
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == '__main__':
    print("开始测试发网检测API...")
    test_hairnet_detection()
    print("测试完成")
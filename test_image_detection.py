#!/usr/bin/env python3
"""
测试图像检测功能的脚本
"""

import requests
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def create_test_image():
    """创建一个测试图像，包含简单的人形轮廓"""
    # 创建一个640x480的白色背景图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # 绘制一个简单的人形轮廓（矩形代表身体）
    # 头部
    cv2.circle(img, (320, 120), 40, (0, 0, 0), 2)
    # 身体
    cv2.rectangle(img, (280, 160), (360, 320), (0, 0, 0), 2)
    # 手臂
    cv2.line(img, (280, 200), (240, 260), (0, 0, 0), 2)
    cv2.line(img, (360, 200), (400, 260), (0, 0, 0), 2)
    # 腿
    cv2.line(img, (300, 320), (280, 400), (0, 0, 0), 2)
    cv2.line(img, (340, 320), (360, 400), (0, 0, 0), 2)
    
    return img

def test_comprehensive_detection():
    """测试综合检测API"""
    # 创建测试图像
    test_img = create_test_image()
    
    # 将图像编码为JPEG格式
    _, buffer = cv2.imencode('.jpg', test_img)
    
    # 准备文件数据
    files = {
        'file': ('test_image.jpg', BytesIO(buffer.tobytes()), 'image/jpeg')
    }
    
    try:
        # 发送请求到综合检测API
        response = requests.post(
            'http://localhost:8000/api/v1/detect/comprehensive',
            files=files,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("检测成功!")
            print(f"检测到的人数: {result.get('total_persons', 0)}")
            print(f"统计信息: {result.get('statistics', {})}")
            print(f"处理时间: {result.get('processing_time', {})}")
        else:
            print(f"检测失败: {response.text}")
            
    except Exception as e:
        print(f"请求失败: {e}")

def test_with_real_image():
    """使用真实图像测试"""
    # 创建一个更大的测试图像
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 240  # 浅灰色背景
    
    # 绘制更详细的人形
    # 人物1
    cv2.circle(img, (300, 150), 60, (100, 100, 100), -1)  # 头部
    cv2.rectangle(img, (240, 210), (360, 500), (100, 100, 100), -1)  # 身体
    cv2.rectangle(img, (200, 250), (240, 400), (100, 100, 100), -1)  # 左臂
    cv2.rectangle(img, (360, 250), (400, 400), (100, 100, 100), -1)  # 右臂
    cv2.rectangle(img, (260, 500), (300, 700), (100, 100, 100), -1)  # 左腿
    cv2.rectangle(img, (320, 500), (360, 700), (100, 100, 100), -1)  # 右腿
    
    # 人物2
    cv2.circle(img, (800, 180), 70, (80, 80, 80), -1)  # 头部
    cv2.rectangle(img, (730, 250), (870, 550), (80, 80, 80), -1)  # 身体
    cv2.rectangle(img, (680, 300), (730, 450), (80, 80, 80), -1)  # 左臂
    cv2.rectangle(img, (870, 300), (920, 450), (80, 80, 80), -1)  # 右臂
    cv2.rectangle(img, (750, 550), (790, 750), (80, 80, 80), -1)  # 左腿
    cv2.rectangle(img, (810, 550), (850, 750), (80, 80, 80), -1)  # 右腿
    
    # 保存测试图像
    cv2.imwrite('test_detection_image.jpg', img)
    print("已创建测试图像: test_detection_image.jpg")
    
    # 将图像编码为JPEG格式
    _, buffer = cv2.imencode('.jpg', img)
    
    # 准备文件数据
    files = {
        'file': ('test_detection_image.jpg', BytesIO(buffer.tobytes()), 'image/jpeg')
    }
    
    try:
        # 发送请求到综合检测API
        response = requests.post(
            'http://localhost:8000/api/v1/detect/comprehensive',
            files=files,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("检测成功!")
            print(f"检测到的人数: {result.get('total_persons', 0)}")
            print(f"统计信息: {result.get('statistics', {})}")
            print(f"处理时间: {result.get('processing_time', {})}")
            
            # 如果有检测结果图像，保存它
            if 'annotated_image' in result:
                img_data = base64.b64decode(result['annotated_image'])
                with open('detection_result.jpg', 'wb') as f:
                    f.write(img_data)
                print("检测结果图像已保存为: detection_result.jpg")
        else:
            print(f"检测失败: {response.text}")
            
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    print("=== 测试综合检测API ===")
    print("\n1. 测试简单图像:")
    test_comprehensive_detection()
    
    print("\n2. 测试真实图像:")
    test_with_real_image()
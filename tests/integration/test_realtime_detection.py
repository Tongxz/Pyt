#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时检测测试脚本
用于验证发网检测在实时场景下的准确性
"""

import requests
import numpy as np
from PIL import Image, ImageDraw
import io
import json

from pathlib import Path
def create_realistic_person_image(with_hairnet=False):
    """
    创建更真实的人物图像用于测试
    
    Args:
        with_hairnet: 是否包含发网
        
    Returns:
        PIL.Image: 生成的图像
    """
    # 创建一个更大的图像
    img = Image.new('RGB', (400, 600), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # 绘制人体轮廓
    # 头部
    head_center = (200, 100)
    head_radius = 60
    draw.ellipse([
        head_center[0] - head_radius, head_center[1] - head_radius,
        head_center[0] + head_radius, head_center[1] + head_radius
    ], fill='peachpuff', outline='black', width=2)
    
    # 身体
    draw.rectangle([170, 160, 230, 400], fill='blue', outline='black', width=2)
    
    # 手臂
    draw.rectangle([130, 180, 170, 350], fill='blue', outline='black', width=2)
    draw.rectangle([230, 180, 270, 350], fill='blue', outline='black', width=2)
    
    # 腿部
    draw.rectangle([175, 400, 200, 550], fill='darkblue', outline='black', width=2)
    draw.rectangle([200, 400, 225, 550], fill='darkblue', outline='black', width=2)
    
    # 头发
    draw.ellipse([
        head_center[0] - head_radius + 5, head_center[1] - head_radius + 5,
        head_center[0] + head_radius - 5, head_center[1] - head_radius + 30
    ], fill='brown', outline='black', width=1)
    
    if with_hairnet:
        # 绘制发网 - 使用更明显的网格模式
        net_color = 'white'
        # 水平线
        for y in range(head_center[1] - head_radius + 10, head_center[1] - head_radius + 35, 8):
            draw.line([
                head_center[0] - head_radius + 10, y,
                head_center[0] + head_radius - 10, y
            ], fill=net_color, width=2)
        
        # 垂直线
        for x in range(head_center[0] - head_radius + 10, head_center[0] + head_radius - 10, 8):
            draw.line([
                x, head_center[1] - head_radius + 10,
                x, head_center[1] - head_radius + 35
            ], fill=net_color, width=2)
        
        # 添加一些对角线增强网状效果
        for i in range(5):
            x_start = head_center[0] - head_radius + 15 + i * 10
            draw.line([
                x_start, head_center[1] - head_radius + 10,
                x_start + 15, head_center[1] - head_radius + 25
            ], fill=net_color, width=1)
    
    # 面部特征
    # 眼睛
    draw.ellipse([185, 85, 195, 95], fill='black')
    draw.ellipse([205, 85, 215, 95], fill='black')
    
    # 鼻子
    draw.line([200, 95, 200, 105], fill='black', width=2)
    
    # 嘴巴
    draw.arc([190, 105, 210, 115], 0, 180, fill='black', width=2)
    
    return img

def test_detection_api(image, test_name):
    """
    测试检测API
    
    Args:
        image: PIL图像
        test_name: 测试名称
    """
    print(f"\n=== {test_name} ===")
    
    # 将图像转换为字节
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    try:
        # 发送请求到API
        response = requests.post(
            'http://localhost:8000/api/v1/detect/hairnet',
            files={'file': ('test.jpg', img_byte_arr, 'image/jpeg')},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            detections = result.get('detections', {})
            
            total_persons = detections.get('total_persons', 0)
            persons_with_hairnet = detections.get('persons_with_hairnet', 0)
            compliance_rate = detections.get('compliance_rate', 0.0)
            persons = detections.get('persons', [])
            
            print(f"检测到 {total_persons} 个人")
            print(f"佩戴发网: {persons_with_hairnet} 人")
            print(f"合规率: {compliance_rate:.2f}%")
            
            for i, person in enumerate(persons, 1):
                print(f"  人员{i}: 发网={person.get('has_hairnet', False)}, 置信度={person.get('confidence', 0.0):.3f}")
                
        else:
            print(f"API请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
    except Exception as e:
        print(f"处理异常: {e}")

def main():
    """
    主测试函数
    """
    print("开始实时检测准确性测试...")
    
    # 测试1: 不带发网的真实场景
    print("\n创建不带发网的测试图像...")
    img_no_hairnet = create_realistic_person_image(with_hairnet=False)
    test_detection_api(img_no_hairnet, "测试1: 不带发网的人员")
    
    # 测试2: 带发网的真实场景
    print("\n创建带发网的测试图像...")
    img_with_hairnet = create_realistic_person_image(with_hairnet=True)
    test_detection_api(img_with_hairnet, "测试2: 带发网的人员")
    
    print("\n实时检测测试完成")
    print("\n注意: 如果不带发网的人员被误检为带发网，说明检测阈值需要进一步调整")

if __name__ == "__main__":
    main()
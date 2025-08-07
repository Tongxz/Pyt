#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化检测管道的性能和功能
"""

import requests
import time
import json
from PIL import Image, ImageDraw
import io
import numpy as np

def create_test_image_with_person(width=800, height=600):
    """创建包含人形轮廓的测试图像"""
    # 创建白色背景
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # 绘制一个简单的人形轮廓
    center_x, center_y = width // 2, height // 2
    
    # 头部 (圆形)
    head_radius = 40
    draw.ellipse([
        center_x - head_radius, center_y - 150,
        center_x + head_radius, center_y - 70
    ], fill='lightblue', outline='black', width=2)
    
    # 身体 (矩形)
    body_width, body_height = 60, 120
    draw.rectangle([
        center_x - body_width//2, center_y - 70,
        center_x + body_width//2, center_y + 50
    ], fill='lightgreen', outline='black', width=2)
    
    # 手臂
    arm_length = 80
    # 左臂
    draw.line([
        center_x - body_width//2, center_y - 40,
        center_x - body_width//2 - arm_length, center_y
    ], fill='brown', width=8)
    # 右臂
    draw.line([
        center_x + body_width//2, center_y - 40,
        center_x + body_width//2 + arm_length, center_y
    ], fill='brown', width=8)
    
    # 腿部
    leg_length = 100
    # 左腿
    draw.line([
        center_x - 20, center_y + 50,
        center_x - 20, center_y + 50 + leg_length
    ], fill='blue', width=12)
    # 右腿
    draw.line([
        center_x + 20, center_y + 50,
        center_x + 20, center_y + 50 + leg_length
    ], fill='blue', width=12)
    
    return img

def download_real_image(url, filename):
    """下载真实图像"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"成功下载图像: {filename}")
        return True
    except Exception as e:
        print(f"下载图像失败: {e}")
        return False

def test_detection_performance(image_path, test_name):
    """测试检测性能"""
    url = "http://localhost:8000/api/v1/detect/comprehensive"
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f, 'image/jpeg')}
            data = {'record_process': 'false'}
            
            response = requests.post(url, files=files, data=data, timeout=30)
        
        # 记录结束时间
        end_time = time.time()
        request_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n=== {test_name} 测试结果 ===")
            print(f"请求时间: {request_time:.3f}秒")
            print(f"检测到的人数: {result.get('total_persons', 0)}")
            
            # 统计信息
            stats = result.get('statistics', {})
            print(f"发网合规率: {stats.get('hairnet_compliance_rate', 0):.2%}")
            print(f"洗手率: {stats.get('handwash_rate', 0):.2%}")
            print(f"消毒率: {stats.get('sanitize_rate', 0):.2%}")
            
            # 处理时间
            processing_time = result.get('processing_time', {})
            print(f"\n处理时间详情:")
            print(f"  人体检测: {processing_time.get('detection_time', 0):.3f}秒")
            print(f"  发网检测: {processing_time.get('hairnet_time', 0):.3f}秒")
            print(f"  行为检测: {processing_time.get('handwash_time', 0):.3f}秒")
            print(f"  总处理时间: {processing_time.get('total_time', 0):.3f}秒")
            
            # 优化统计信息
            optimization_stats = result.get('optimization_stats', {})
            if optimization_stats:
                print(f"\n优化统计:")
                print(f"  缓存命中率: {optimization_stats.get('cache_hit_rate', 0):.2%}")
                print(f"  缓存大小: {optimization_stats.get('cache_size', 0)}")
                print(f"  总检测次数: {optimization_stats.get('total_detections', 0)}")
                print(f"  缓存命中次数: {optimization_stats.get('cache_hits', 0)}")
            
            return True
        else:
            print(f"请求失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def performance_comparison_test():
    """性能对比测试"""
    print("\n=== 性能对比测试 ===")
    
    # 使用tests目录下的现有测试图像
    test_cases = [
        ('tests/fixtures/images/person/test_person.png', '测试图像'),
        ('tests/fixtures/images/hairnet/7月23日.png', '发网测试图像')
    ]
    
    for image_path, test_name in test_cases:
        print(f"\n开始测试: {test_name}")
        
        # 进行多次测试
        times = []
        for i in range(3):
            print(f"第 {i+1} 次测试...")
            start = time.time()
            success = test_detection_performance(image_path, f"{test_name}-第{i+1}次")
            if success:
                times.append(time.time() - start)
            time.sleep(1)  # 间隔1秒
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"\n{test_name} 平均响应时间: {avg_time:.3f}秒")
            print(f"最快: {min(times):.3f}秒, 最慢: {max(times):.3f}秒")

def main():
    """主测试函数"""
    print("开始测试优化检测管道...")
    
    # 检查服务器是否运行
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("服务器未正常运行，请先启动服务器")
            return
    except Exception as e:
        print(f"无法连接到服务器: {e}")
        return
    
    print("服务器连接正常，开始性能测试...")
    
    # 运行性能对比测试
    performance_comparison_test()
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()
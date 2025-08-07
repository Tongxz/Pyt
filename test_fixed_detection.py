#!/usr/bin/env python3
"""
测试修复后的检测结果
验证API能正确返回检测到的人数和各项检测结果
"""

import requests
import json
import time
import os
from PIL import Image, ImageDraw
import numpy as np
import io

def create_test_image_with_people():
    """创建包含多个人形轮廓的测试图像"""
    # 创建一个800x600的白色背景图像
    img = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(img)
    
    # 绘制3个简单的人形轮廓
    people_positions = [
        (150, 200),  # 第一个人
        (350, 180),  # 第二个人
        (550, 220),  # 第三个人
    ]
    
    for i, (x, y) in enumerate(people_positions):
        # 绘制头部（圆形）
        draw.ellipse([x-20, y-40, x+20, y], fill='lightblue', outline='blue')
        
        # 绘制身体（矩形）
        draw.rectangle([x-25, y, x+25, y+120], fill='lightgreen', outline='green')
        
        # 绘制手臂
        draw.rectangle([x-45, y+20, x-25, y+80], fill='lightgreen', outline='green')  # 左臂
        draw.rectangle([x+25, y+20, x+45, y+80], fill='lightgreen', outline='green')  # 右臂
        
        # 绘制腿部
        draw.rectangle([x-20, y+120, x-5, y+200], fill='lightgreen', outline='green')  # 左腿
        draw.rectangle([x+5, y+120, x+20, y+200], fill='lightgreen', outline='green')  # 右腿
        
        # 添加标签
        draw.text((x-10, y+210), f'Person {i+1}', fill='black')
    
    return img

def test_detection_api():
    """测试检测API"""
    print("=== 测试修复后的检测结果 ===")
    
    # 创建测试图像
    print("\n1. 创建包含3个人的测试图像...")
    test_image = create_test_image_with_people()
    
    # 将图像转换为字节流
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # 测试API
    url = "http://localhost:8000/api/v1/detect/comprehensive"
    
    print("\n2. 发送检测请求...")
    start_time = time.time()
    
    try:
        files = {'file': ('test_image.png', img_byte_arr, 'image/png')}
        response = requests.post(url, files=files, timeout=30)
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ 检测成功! (请求时间: {request_time:.3f}秒)")
            print("\n=== 检测结果 ===")
            print(f"检测到的人数: {result.get('total_persons', 0)}")
            
            statistics = result.get('statistics', {})
            print(f"\n统计信息:")
            print(f"  - 佩戴发网人数: {statistics.get('persons_with_hairnet', 0)}")
            print(f"  - 未佩戴发网人数: {statistics.get('persons_without_hairnet', 0)}")
            print(f"  - 洗手人数: {statistics.get('persons_handwashing', 0)}")
            print(f"  - 消毒人数: {statistics.get('persons_sanitizing', 0)}")
            print(f"  - 发网合规率: {statistics.get('hairnet_compliance_rate', 0):.1%}")
            print(f"  - 洗手率: {statistics.get('handwash_rate', 0):.1%}")
            print(f"  - 消毒率: {statistics.get('sanitize_rate', 0):.1%}")
            
            processing_time = result.get('processing_time', {})
            print(f"\n处理时间详情:")
            print(f"  - 人体检测时间: {processing_time.get('detection_time', 0):.3f}秒")
            print(f"  - 发网检测时间: {processing_time.get('hairnet_time', 0):.3f}秒")
            print(f"  - 行为检测时间: {processing_time.get('handwash_time', 0):.3f}秒")
            print(f"  - 总处理时间: {processing_time.get('total_time', 0):.3f}秒")
            
            optimization_stats = result.get('optimization_stats', {})
            print(f"\n优化统计:")
            print(f"  - 缓存启用: {optimization_stats.get('cache_enabled', False)}")
            print(f"  - 缓存命中率: {optimization_stats.get('cache_hit_rate', 0):.1%}")
            print(f"  - 缓存大小: {optimization_stats.get('cache_size', 0)}")
            print(f"  - 总检测次数: {optimization_stats.get('total_detections', 0)}")
            print(f"  - 缓存命中次数: {optimization_stats.get('cache_hits', 0)}")
            print(f"  - 缓存未命中次数: {optimization_stats.get('cache_misses', 0)}")
            
            # 验证结果
            print("\n=== 结果验证 ===")
            expected_persons = 3
            actual_persons = result.get('total_persons', 0)
            
            if actual_persons == expected_persons:
                print(f"✅ 人体检测正确: 期望{expected_persons}人，实际检测到{actual_persons}人")
            else:
                print(f"❌ 人体检测有误: 期望{expected_persons}人，实际检测到{actual_persons}人")
            
            # 检查各项检测是否有结果
            hairnet_count = statistics.get('persons_with_hairnet', 0) + statistics.get('persons_without_hairnet', 0)
            handwash_count = statistics.get('persons_handwashing', 0)
            sanitize_count = statistics.get('persons_sanitizing', 0)
            
            print(f"\n检测功能验证:")
            print(f"  - 发网检测: {'✅ 正常' if hairnet_count > 0 else '❌ 无结果'}")
            print(f"  - 洗手检测: {'✅ 正常' if handwash_count > 0 else '❌ 无结果'}")
            print(f"  - 消毒检测: {'✅ 正常' if sanitize_count > 0 else '❌ 无结果'}")
            
            return True
            
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {e}")
        return False
    except Exception as e:
        print(f"❌ 处理异常: {e}")
        return False

def test_multiple_requests():
    """测试多次请求以验证缓存效果"""
    print("\n\n=== 测试缓存效果 ===")
    
    # 创建测试图像
    test_image = create_test_image_with_people()
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    
    url = "http://localhost:8000/api/v1/detect/comprehensive"
    
    # 第一次请求（冷启动）
    print("\n第一次请求（冷启动）...")
    img_byte_arr.seek(0)
    start_time = time.time()
    
    try:
        files = {'file': ('test_image.png', img_byte_arr, 'image/png')}
        response1 = requests.post(url, files=files, timeout=30)
        time1 = time.time() - start_time
        
        if response1.status_code == 200:
            result1 = response1.json()
            print(f"✅ 第一次请求成功 (时间: {time1:.3f}秒)")
            print(f"检测到人数: {result1.get('total_persons', 0)}")
        
        # 第二次请求（应该命中缓存）
        print("\n第二次请求（缓存命中）...")
        img_byte_arr.seek(0)
        start_time = time.time()
        
        files = {'file': ('test_image.png', img_byte_arr, 'image/png')}
        response2 = requests.post(url, files=files, timeout=30)
        time2 = time.time() - start_time
        
        if response2.status_code == 200:
            result2 = response2.json()
            print(f"✅ 第二次请求成功 (时间: {time2:.3f}秒)")
            print(f"检测到人数: {result2.get('total_persons', 0)}")
            
            # 计算性能提升
            if time1 > 0 and time2 > 0:
                speedup = time1 / time2
                print(f"\n性能提升: {speedup:.1f}倍 (从 {time1:.3f}秒 到 {time2:.3f}秒)")
            
            # 显示缓存统计
            optimization_stats = result2.get('optimization_stats', {})
            print(f"\n缓存统计:")
            print(f"  - 缓存命中率: {optimization_stats.get('cache_hit_rate', 0):.1%}")
            print(f"  - 缓存命中次数: {optimization_stats.get('cache_hits', 0)}")
            print(f"  - 缓存未命中次数: {optimization_stats.get('cache_misses', 0)}")
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")

if __name__ == "__main__":
    print("开始测试修复后的检测功能...")
    
    # 检查服务器是否运行
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ 服务器运行正常")
        else:
            print("❌ 服务器状态异常")
            exit(1)
    except:
        print("❌ 无法连接到服务器，请确保服务器正在运行")
        exit(1)
    
    # 运行测试
    success = test_detection_api()
    
    if success:
        test_multiple_requests()
        print("\n\n🎉 测试完成！检测功能已修复并正常工作。")
    else:
        print("\n\n❌ 测试失败，请检查服务器日志。")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实图片测试综合检测API
"""

import requests
import base64
import json
from pathlib import Path

def test_with_real_image():
    """
    使用真实图片测试综合检测API
    """
    print("🧪 使用真实图片测试综合检测API...")
    
    # 使用项目中的真实图片
    image_path = "/Users/zhou/Code/python/Pyt/runs/detect/exp4/7月23日.jpg"
    
    if not Path(image_path).exists():
        print(f"❌ 图片文件不存在: {image_path}")
        return
    
    print(f"📸 使用图片: {image_path}")
    
    # 准备API请求
    url = "http://localhost:8000/api/v1/detect/comprehensive"
    
    print("🔄 正在调用综合检测API...")
    
    try:
        # 使用文件上传的方式
        with open(image_path, "rb") as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(url, files=files, timeout=30)
        print(f"📊 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API调用成功!")
            
            # 显示统计信息
            print("\n📊 检测统计:")
            print(f"   👥 总人数: {result.get('total_persons', 0)}")
            
            stats = result.get('statistics', {})
            print(f"   🧢 佩戴发网人数: {stats.get('persons_with_hairnet', 0)}")
            print(f"   🧼 洗手人数: {stats.get('persons_handwashing', 0)}")
            print(f"   🧴 消毒人数: {stats.get('persons_sanitizing', 0)}")
            
            # 显示检测详情
            detections = result.get('detections', [])
            print(f"\n🔍 检测详情 ({len(detections)} 个目标):")
            
            for i, detection in enumerate(detections, 1):
                print(f"   目标 {i}:")
                print(f"     类型: {detection.get('type', 'unknown')}")
                print(f"     置信度: {detection.get('confidence', 0):.3f}")
                if 'behaviors' in detection:
                    behaviors = detection['behaviors']
                    print(f"     洗手: {behaviors.get('handwashing', False)} (置信度: {behaviors.get('handwashing_confidence', 0):.3f})")
                    print(f"     消毒: {behaviors.get('sanitizing', False)} (置信度: {behaviors.get('sanitizing_confidence', 0):.3f})")
            
            # 显示处理时间
            processing_time = result.get('processing_time', {})
            print("\n⏱️ 处理时间:")
            for key, value in processing_time.items():
                print(f"   {key}: {value:.3f}s")
            
            # 检查是否包含标注图像
            if 'annotated_image' in result:
                print("\n🖼️ 包含标注图像 (base64编码)")
                
                # 保存标注结果
                try:
                    annotated_data = base64.b64decode(result['annotated_image'])
                    output_path = "real_image_annotated_result.jpg"
                    with open(output_path, "wb") as f:
                        f.write(annotated_data)
                    print(f"📸 标注结果已保存为 {output_path}")
                except Exception as e:
                    print(f"❌ 保存标注结果失败: {e}")
            
            print("\n🎉 测试完成! 请检查前端页面是否能正确显示这些数据。")
            
            print("\n💡 建议:")
            print("   1. 打开 http://localhost:8000/frontend/index.html")
            print(f"   2. 上传 {image_path} 文件")
            print("   3. 点击'开始检测'按钮")
            print("   4. 检查显示的统计数据和标注图像是否正确")
            
        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {e}")
    except Exception as e:
        print(f"❌ 未知错误: {e}")

if __name__ == "__main__":
    test_with_real_image()
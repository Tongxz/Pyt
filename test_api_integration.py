#!/usr/bin/env python3
"""
测试API集成和手部检测功能
"""

import requests
import json
from pathlib import Path

def test_comprehensive_detection():
    """测试综合检测API"""
    url = "http://localhost:8000/api/v1/detect/comprehensive"
    
    # 查找测试图像
    test_image_path = "realistic_test_image.jpg"
    if not Path(test_image_path).exists():
        print(f"测试图像不存在: {test_image_path}")
        return False
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': (test_image_path, f, 'image/jpeg')}
            data = {'record_process': 'false'}
            
            print(f"正在测试综合检测API: {url}")
            response = requests.post(url, files=files, data=data, timeout=30)
            
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✓ API调用成功")
                print(f"检测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
                
                # 检查是否包含手部检测相关结果
                if 'statistics' in result:
                    stats = result['statistics']
                    print(f"\n=== 检测统计 ===")
                    print(f"洗手人数: {stats.get('persons_handwashing', 0)}")
                    print(f"消毒人数: {stats.get('persons_sanitizing', 0)}")
                    print(f"佩戴发网人数: {stats.get('persons_with_hairnet', 0)}")
                
                return True
            else:
                print(f"✗ API调用失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到API服务器，请确保服务器正在运行")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_image_detection():
    """测试图像检测API"""
    url = "http://localhost:8000/api/v1/detect/image"
    
    test_image_path = "realistic_test_image.jpg"
    if not Path(test_image_path).exists():
        print(f"测试图像不存在: {test_image_path}")
        return False
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': (test_image_path, f, 'image/jpeg')}
            
            print(f"正在测试图像检测API: {url}")
            response = requests.post(url, files=files, timeout=30)
            
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✓ 图像检测API调用成功")
                print(f"检测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return True
            else:
                print(f"✗ 图像检测API调用失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return False
                
    except Exception as e:
        print(f"✗ 图像检测测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== 手部检测API集成测试 ===")
    print()
    
    # 测试综合检测API
    print("1. 测试综合检测API")
    comprehensive_success = test_comprehensive_detection()
    print()
    
    # 测试图像检测API
    print("2. 测试图像检测API")
    image_success = test_image_detection()
    print()
    
    # 总结
    print("=== 测试总结 ===")
    print(f"综合检测API: {'✓ 通过' if comprehensive_success else '✗ 失败'}")
    print(f"图像检测API: {'✓ 通过' if image_success else '✗ 失败'}")
    
    if comprehensive_success and image_success:
        print("\n🎉 所有API测试通过！手部检测器已成功集成到web端。")
    else:
        print("\n⚠️  部分API测试失败，请检查服务器状态和配置。")

if __name__ == "__main__":
    main()
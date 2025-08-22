#!/usr/bin/env python3
"""
简单的API连接测试
"""

import requests
import json

def test_health_check():
    """测试健康检查端点"""
    try:
        url = "http://localhost:8000/health"
        print(f"测试健康检查: {url}")
        response = requests.get(url, timeout=5)
        print(f"响应状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"响应内容: {response.json()}")
            return True
        return False
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_docs():
    """测试API文档端点"""
    try:
        url = "http://localhost:8000/docs"
        print(f"测试API文档: {url}")
        response = requests.get(url, timeout=5)
        print(f"响应状态码: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"API文档测试失败: {e}")
        return False

def main():
    print("=== 简单API连接测试 ===")
    print()
    
    health_ok = test_health_check()
    print()
    
    docs_ok = test_docs()
    print()
    
    print("=== 测试结果 ===")
    print(f"健康检查: {'✓ 通过' if health_ok else '✗ 失败'}")
    print(f"API文档: {'✓ 通过' if docs_ok else '✗ 失败'}")
    
    if health_ok:
        print("\n✓ API服务器运行正常，可以进行手部检测测试")
    else:
        print("\n✗ API服务器连接失败")

if __name__ == "__main__":
    main()
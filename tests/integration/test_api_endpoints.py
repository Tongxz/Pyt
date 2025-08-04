#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API端点集成测试
API Endpoints Integration Tests
"""

import unittest
import requests
import json
import time
import os
import sys
from io import BytesIO
from PIL import Image
import numpy as np

from pathlib import Path
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestAPIEndpoints(unittest.TestCase):
    """API端点集成测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.base_url = "http://localhost:8000"
        cls.timeout = 10
        
        # 等待服务器启动
        cls._wait_for_server()
    
    @classmethod
    def _wait_for_server(cls, max_attempts=30):
        """等待服务器启动"""
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{cls.base_url}/health", timeout=cls.timeout)
                if response.status_code == 200:
                    print(f"服务器已启动，尝试次数: {attempt + 1}")
                    return
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        raise Exception(f"服务器在 {max_attempts} 秒内未启动")
    
    def test_health_check(self):
        """测试健康检查端点"""
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('message', data)
        self.assertIn('detector_ready', data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_api_info(self):
        """测试API信息端点"""
        response = requests.get(f"{self.base_url}/api/info", timeout=self.timeout)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('name', data)
        self.assertIn('version', data)
        self.assertIn('description', data)
        self.assertIn('endpoints', data)
        
        # 检查端点列表
        endpoints = data['endpoints']
        self.assertIsInstance(endpoints, list)
        self.assertGreater(len(endpoints), 0)
    
    def test_detect_image_endpoint(self):
        """测试图像检测端点"""
        # 创建测试图像
        test_image = self._create_test_image()
        
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        data = {
            'confidence_threshold': 0.5,
            'visualize': True
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/detect/image",
            files=files,
            data=data,
            timeout=self.timeout
        )
        
        # 检查响应状态
        if response.status_code == 404:
            self.skipTest("图像检测端点未实现")
        
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertIn('detections', result)
        self.assertIn('total_persons', result)
        self.assertIn('processing_time', result)
    
    def test_detect_hairnet_endpoint(self):
        """测试发网检测端点"""
        # 创建测试图像
        test_image = self._create_test_image()
        
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        data = {
            'confidence_threshold': 0.7,
            'visualize': True
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/detect/hairnet",
            files=files,
            data=data,
            timeout=self.timeout
        )
        
        # 检查响应状态
        if response.status_code == 404:
            self.skipTest("发网检测端点未实现")
        
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertIn('total_persons', result)
        self.assertIn('persons_with_hairnet', result)
        self.assertIn('persons_without_hairnet', result)
        self.assertIn('compliance_rate', result)
        self.assertIn('detections', result)
    
    def test_detect_video_endpoint(self):
        """测试视频检测端点"""
        # 创建测试视频文件（简单的图像序列）
        test_video = self._create_test_video()
        
        files = {'file': ('test.mp4', test_video, 'video/mp4')}
        data = {
            'confidence_threshold': 0.5,
            'frame_skip': 5
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/detect/video",
            files=files,
            data=data,
            timeout=30  # 视频处理需要更长时间
        )
        
        # 检查响应状态
        if response.status_code == 404:
            self.skipTest("视频检测端点未实现")
        
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertIn('total_frames', result)
        self.assertIn('processed_frames', result)
        self.assertIn('detections', result)
        self.assertIn('processing_time', result)
    
    def test_statistics_endpoint(self):
        """测试统计信息端点"""
        response = requests.get(f"{self.base_url}/api/statistics", timeout=self.timeout)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('total_detections', data)
        self.assertIn('total_persons_detected', data)
        self.assertIn('average_compliance_rate', data)
        self.assertIn('last_updated', data)
    
    def test_statistics_realtime_endpoint(self):
        """测试实时统计信息端点"""
        response = requests.get(f"{self.base_url}/api/statistics/realtime", timeout=self.timeout)
        
        # 检查响应状态
        if response.status_code == 404:
            self.skipTest("实时统计端点未实现")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('current_persons', data)
        self.assertIn('current_compliance_rate', data)
        self.assertIn('timestamp', data)
    
    def test_statistics_with_date_range(self):
        """测试带日期范围的统计信息端点"""
        params = {
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }
        
        response = requests.get(
            f"{self.base_url}/api/statistics",
            params=params,
            timeout=self.timeout
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('total_detections', data)
        self.assertIn('date_range', data)
    
    def test_invalid_image_upload(self):
        """测试无效图像上传"""
        # 上传非图像文件
        files = {'file': ('test.txt', b'not an image', 'text/plain')}
        
        response = requests.post(
            f"{self.base_url}/api/v1/detect/image",
            files=files,
            timeout=self.timeout
        )
        
        # 应该返回错误状态码
        self.assertIn(response.status_code, [400, 422, 500])
    
    def test_missing_file_upload(self):
        """测试缺少文件的上传请求"""
        response = requests.post(
            f"{self.base_url}/api/v1/detect/image",
            timeout=self.timeout
        )
        
        # 应该返回错误状态码
        self.assertIn(response.status_code, [400, 422])
    
    def test_invalid_parameters(self):
        """测试无效参数"""
        test_image = self._create_test_image()
        
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        data = {
            'confidence_threshold': 'invalid',  # 无效的置信度值
            'visualize': 'not_boolean'  # 无效的布尔值
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/detect/image",
            files=files,
            data=data,
            timeout=self.timeout
        )
        
        # 应该返回错误状态码或使用默认值
        self.assertIn(response.status_code, [200, 400, 422])
    
    def test_cors_headers(self):
        """测试CORS头部"""
        response = requests.options(f"{self.base_url}/health", timeout=self.timeout)
        
        # 检查CORS头部
        headers = response.headers
        self.assertIn('Access-Control-Allow-Origin', headers)
    
    def test_api_response_format(self):
        """测试API响应格式"""
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        
        # 检查Content-Type
        self.assertEqual(response.headers.get('Content-Type'), 'application/json')
        
        # 检查JSON格式
        try:
            data = response.json()
            self.assertIsInstance(data, dict)
        except json.JSONDecodeError:
            self.fail("响应不是有效的JSON格式")
    
    def test_error_handling(self):
        """测试错误处理"""
        # 访问不存在的端点
        response = requests.get(f"{self.base_url}/api/nonexistent", timeout=self.timeout)
        
        self.assertEqual(response.status_code, 404)
    
    def _create_test_image(self, width=640, height=480):
        """创建测试图像"""
        # 创建随机图像
        image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # 转换为字节流
        img_buffer = BytesIO()
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        return img_buffer
    
    def _create_test_video(self):
        """创建测试视频文件（模拟）"""
        # 创建一个简单的字节流作为模拟视频
        # 在实际测试中，这里应该是真实的视频文件
        video_data = b'fake video data for testing'
        return BytesIO(video_data)
    
    def test_concurrent_requests(self):
        """测试并发请求"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
                results.put(response.status_code)
            except Exception as e:
                results.put(str(e))
        
        # 创建多个并发请求
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查结果
        success_count = 0
        while not results.empty():
            result = results.get()
            if result == 200:
                success_count += 1
        
        # 至少应该有一些成功的请求
        self.assertGreater(success_count, 0)

if __name__ == '__main__':
    unittest.main()
"""API接口测试模块.

测试FastAPI应用程序的各个端点.
"""
import io
import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.services.detection_service import get_optimized_pipeline, get_hairnet_pipeline
from src.services.region_service import get_region_service


class TestAPIEndpoints:
    """API端点测试类."""

    def setup_method(self):
        """测试方法设置."""
        self.client = TestClient(app)
        # 清除之前的依赖覆盖
        app.dependency_overrides.clear()
    
    def teardown_method(self):
        """测试方法清理."""
        # 清除依赖覆盖
        app.dependency_overrides.clear()

    def test_health_check(self):
        """测试健康检查端点."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_root_redirect(self):
        """测试根路径重定向."""
        response = self.client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/frontend/index.html"

    def test_detect_image_endpoint(self):
        """测试图像检测端点."""
        # 模拟优化管道
        mock_pipeline = Mock()
        app.dependency_overrides[get_optimized_pipeline] = lambda: mock_pipeline

        # 创建测试图像文件
        test_image_data = b"fake_image_data"
        files = {"file": ("test.jpg", io.BytesIO(test_image_data), "image/jpeg")}

        response = self.client.post("/api/v1/detect/image", files=files)
        
        assert response.status_code == 200
        result = response.json()
        assert result["filename"] == "test.jpg"
        assert result["detection_type"] == "image"
        assert "results" in result
        assert result["status"] == "success"

    def test_detect_hairnet_endpoint(self):
        """测试发网检测端点."""
        # 模拟发网检测管道
        mock_hairnet = Mock()
        app.dependency_overrides[get_hairnet_pipeline] = lambda: mock_hairnet

        # 创建测试图像文件
        test_image_data = b"fake_image_data"
        files = {"file": ("test.jpg", io.BytesIO(test_image_data), "image/jpeg")}

        response = self.client.post("/api/v1/detect/hairnet", files=files)
        
        assert response.status_code == 200
        result = response.json()
        assert result["filename"] == "test.jpg"
        assert result["detection_type"] == "hairnet"
        assert "results" in result
        assert result["status"] == "success"

    def test_detect_image_no_file(self):
        """测试图像检测端点无文件情况."""
        response = self.client.post("/api/v1/detect/image")
        assert response.status_code == 422  # Unprocessable Entity

    def test_detect_hairnet_no_file(self):
        """测试发网检测端点无文件情况."""
        response = self.client.post("/api/v1/detect/hairnet")
        assert response.status_code == 422  # Unprocessable Entity

    def test_detect_image_no_pipeline(self):
        """测试图像检测端点管道未初始化情况."""
        # 模拟管道未初始化
        app.dependency_overrides[get_optimized_pipeline] = lambda: None

        test_image_data = b"fake_image_data"
        files = {"file": ("test.jpg", io.BytesIO(test_image_data), "image/jpeg")}

        response = self.client.post("/api/v1/detect/image", files=files)
        
        assert response.status_code == 500
        assert "检测服务未初始化" in response.json()["detail"]

    def test_detect_hairnet_no_pipeline(self):
        """测试发网检测端点管道未初始化情况."""
        # 模拟管道未初始化
        app.dependency_overrides[get_hairnet_pipeline] = lambda: None

        test_image_data = b"fake_image_data"
        files = {"file": ("test.jpg", io.BytesIO(test_image_data), "image/jpeg")}

        response = self.client.post("/api/v1/detect/hairnet", files=files)
        
        assert response.status_code == 500
        assert "发网检测服务未初始化" in response.json()["detail"]

    def test_realtime_statistics_endpoint(self):
        """测试实时统计端点."""
        # 模拟区域服务
        mock_region_service = Mock()
        app.dependency_overrides[get_region_service] = lambda: mock_region_service

        response = self.client.get("/api/v1/statistics/realtime")
        
        assert response.status_code == 200
        result = response.json()
        
        # 验证返回数据结构
        assert "timestamp" in result
        assert "system_status" in result
        assert "detection_stats" in result
        assert "region_stats" in result
        assert "performance_metrics" in result
        assert "alerts" in result
        
        # 验证检测统计数据结构
        detection_stats = result["detection_stats"]
        assert "total_detections_today" in detection_stats
        assert "handwashing_detections" in detection_stats
        assert "disinfection_detections" in detection_stats
        assert "hairnet_detections" in detection_stats
        assert "violation_count" in detection_stats

    def test_realtime_statistics_no_region_service(self):
        """测试实时统计端点无区域服务情况."""
        app.dependency_overrides[get_region_service] = lambda: None
        response = self.client.get("/api/v1/statistics/realtime")
        
        assert response.status_code == 200
        result = response.json()
        assert result["system_status"] == "active"

    def test_statistics_endpoint(self):
        """测试统计信息端点."""
        mock_region_service = Mock()
        app.dependency_overrides[get_region_service] = lambda: mock_region_service
        
        response = self.client.get("/api/v1/statistics")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_violations_endpoint(self):
        """测试违规记录端点."""
        mock_region_service = Mock()
        app.dependency_overrides[get_region_service] = lambda: mock_region_service
        
        response = self.client.get("/api/v1/violations")
        assert response.status_code == 200
        assert "message" in response.json()


class TestAPIErrorHandling:
    """API错误处理测试类."""

    def setup_method(self):
        """测试方法设置."""
        self.client = TestClient(app)

    def test_invalid_endpoint(self):
        """测试无效端点."""
        response = self.client.get("/api/v1/invalid")
        assert response.status_code == 404

    def test_method_not_allowed(self):
        """测试不允许的HTTP方法."""
        response = self.client.put("/health")
        assert response.status_code == 405
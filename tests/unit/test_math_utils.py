#!/usr/bin/env python3
"""
数学工具函数单元测试
Math utilities unit tests
"""

try:
    import pytest
except ImportError:
    pytest = None

import math
from pathlib import Path
from typing import List, Tuple

try:
    from src.utils.math_utils import (
        angle_between_points,
        bbox_area,
        bbox_center,
        bbox_iou,
        clamp,
        euclidean_distance,
        manhattan_distance,
        moving_average,
        normalize_angle,
        point_in_polygon,
        smooth_value,
    )
except ImportError:
    # 如果直接运行测试文件
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.utils.math_utils import (
        angle_between_points,
        bbox_area,
        bbox_center,
        bbox_iou,
        clamp,
        euclidean_distance,
        manhattan_distance,
        moving_average,
        normalize_angle,
        point_in_polygon,
        smooth_value,
    )


class TestDistanceFunctions:
    """距离计算函数测试"""

    def test_euclidean_distance(self):
        """测试欧几里得距离计算"""
        # 测试基本情况
        assert euclidean_distance((0, 0), (3, 4)) == 5.0
        assert euclidean_distance((0, 0), (0, 0)) == 0.0
        assert euclidean_distance((1, 1), (1, 1)) == 0.0

        # 测试负坐标
        assert euclidean_distance((-1, -1), (2, 3)) == 5.0

    def test_manhattan_distance(self):
        """测试曼哈顿距离计算"""
        assert manhattan_distance((0, 0), (3, 4)) == 7.0
        assert manhattan_distance((0, 0), (0, 0)) == 0.0
        assert manhattan_distance((1, 1), (1, 1)) == 0.0

        # 测试负坐标
        assert manhattan_distance((-1, -1), (2, 3)) == 7.0


class TestAngleFunctions:
    """角度计算函数测试"""

    def test_angle_between_points(self):
        """测试三点夹角计算"""
        # 测试直角
        angle = angle_between_points((0, 0), (1, 0), (1, 1))
        assert abs(angle - math.pi / 2) < 1e-10

        # 测试平角
        angle = angle_between_points((0, 0), (1, 0), (2, 0))
        assert abs(angle - math.pi) < 1e-10

        # 测试锐角
        angle = angle_between_points((0, 0), (1, 0), (1, 0.5))
        assert 0 < angle <= math.pi / 2

    def test_normalize_angle(self):
        """测试角度标准化"""
        assert abs(normalize_angle(0) - 0) < 1e-10
        assert abs(normalize_angle(2 * math.pi) - 0) < 1e-10
        assert abs(normalize_angle(-math.pi) - math.pi) < 1e-10
        assert abs(normalize_angle(3 * math.pi) - math.pi) < 1e-10


class TestPolygonFunctions:
    """多边形相关函数测试"""

    def test_point_in_polygon(self):
        """测试点在多边形内判断"""
        # 正方形
        square = [(0, 0), (2, 0), (2, 2), (0, 2)]

        # 内部点
        assert point_in_polygon((1, 1), square) == True

        # 外部点
        assert point_in_polygon((3, 3), square) == False
        assert point_in_polygon((-1, 1), square) == False

        # 边界点（可能因实现而异）
        # assert point_in_polygon((0, 1), square) == True

        # 三角形
        triangle = [(0, 0), (2, 0), (1, 2)]
        assert point_in_polygon((1, 0.5), triangle) == True
        assert point_in_polygon((0, 1), triangle) == False


class TestBboxFunctions:
    """边界框相关函数测试"""

    def test_bbox_area(self):
        """测试边界框面积计算"""
        assert bbox_area((0, 0, 2, 2)) == 4.0
        assert bbox_area((1, 1, 3, 4)) == 6.0
        assert bbox_area((0, 0, 0, 0)) == 0.0

        # 测试无效边界框
        assert bbox_area((2, 2, 1, 1)) == 0.0

    def test_bbox_center(self):
        """测试边界框中心点计算"""
        assert bbox_center((0, 0, 2, 2)) == (1.0, 1.0)
        assert bbox_center((1, 1, 3, 5)) == (2.0, 3.0)
        assert bbox_center((0, 0, 0, 0)) == (0.0, 0.0)

    def test_bbox_iou(self):
        """测试边界框IoU计算"""
        # 完全重叠
        assert bbox_iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0

        # 完全不重叠
        assert bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0

        # 部分重叠
        iou = bbox_iou((0, 0, 2, 2), (1, 1, 3, 3))
        expected = 1.0 / 7.0  # 交集面积1，并集面积7
        assert abs(iou - expected) < 1e-10


class TestUtilityFunctions:
    """工具函数测试"""

    def test_smooth_value(self):
        """测试数值平滑"""
        # 默认alpha=0.7
        result = smooth_value(10.0, 20.0)
        expected = 0.7 * 10.0 + 0.3 * 20.0
        assert abs(result - expected) < 1e-10

        # 自定义alpha
        result = smooth_value(10.0, 20.0, alpha=0.5)
        expected = 0.5 * 10.0 + 0.5 * 20.0
        assert abs(result - expected) < 1e-10

    def test_clamp(self):
        """测试数值限制"""
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10
        assert clamp(5.5, 0.0, 10.0) == 5.5

    def test_moving_average(self):
        """测试移动平均"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        # 窗口大小为3
        result = moving_average(values, window_size=3)
        expected = [1.0, 1.5, 2.0, 3.0, 4.0]

        for i, (r, e) in enumerate(zip(result, expected)):
            assert abs(r - e) < 1e-10, f"Index {i}: {r} != {e}"

        # 窗口大小大于数据长度
        result = moving_average([1.0, 2.0], window_size=5)
        assert result == [1.0, 2.0]

        # 空列表
        result = moving_average([], window_size=3)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__])

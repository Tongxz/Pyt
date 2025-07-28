# Mathematical utilities
# 数学工具模块

import math
from typing import List, Tuple, Union, Sequence


def euclidean_distance(point1: Tuple[float, float], 
                      point2: Tuple[float, float]) -> float:
    """
    计算两点间的欧几里得距离
    
    Args:
        point1: 点1 (x, y)
        point2: 点2 (x, y)
    
    Returns:
        欧几里得距离
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def manhattan_distance(point1: Tuple[float, float], 
                      point2: Tuple[float, float]) -> float:
    """
    计算两点间的曼哈顿距离
    
    Args:
        point1: 点1 (x, y)
        point2: 点2 (x, y)
    
    Returns:
        曼哈顿距离
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x2 - x1) + abs(y2 - y1)


def angle_between_points(point1: Tuple[float, float],
                        point2: Tuple[float, float],
                        point3: Tuple[float, float]) -> float:
    """
    计算三点间的夹角 (point2为顶点)
    
    Args:
        point1: 点1 (x, y)
        point2: 点2 (x, y) - 顶点
        point3: 点3 (x, y)
    
    Returns:
        夹角 (弧度)
    """
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    
    # 计算向量
    vec1 = (x1 - x2, y1 - y2)
    vec2 = (x3 - x2, y3 - y2)
    
    # 计算点积和模长
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    magnitude1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    magnitude2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # 计算夹角
    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # 确保在有效范围内
    
    return math.acos(cos_angle)


def point_in_polygon(point: Tuple[float, float], 
                     polygon: Sequence[Tuple[float, float]]) -> bool:
    """
    判断点是否在多边形内 (射线法)
    
    Args:
        point: 测试点 (x, y)
        polygon: 多边形顶点列表 [(x, y), ...]
    
    Returns:
        是否在多边形内
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
                    else:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """
    计算边界框面积
    
    Args:
        bbox: 边界框 (x1, y1, x2, y2)
    
    Returns:
        面积
    """
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    计算边界框中心点
    
    Args:
        bbox: 边界框 (x1, y1, x2, y2)
    
    Returns:
        中心点 (x, y)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_iou(bbox1: Tuple[float, float, float, float],
             bbox2: Tuple[float, float, float, float]) -> float:
    """
    计算两个边界框的IoU
    
    Args:
        bbox1: 边界框1 (x1, y1, x2, y2)
        bbox2: 边界框2 (x1, y1, x2, y2)
    
    Returns:
        IoU值
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算并集
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def smooth_value(current_value: float, 
                new_value: float, 
                alpha: float = 0.7) -> float:
    """
    平滑数值 (指数移动平均)
    
    Args:
        current_value: 当前值
        new_value: 新值
        alpha: 平滑系数 (0-1)
    
    Returns:
        平滑后的值
    """
    return alpha * current_value + (1 - alpha) * new_value


def clamp(value: Union[int, float], 
         min_val: Union[int, float], 
         max_val: Union[int, float]) -> Union[int, float]:
    """
    将值限制在指定范围内
    
    Args:
        value: 输入值
        min_val: 最小值
        max_val: 最大值
    
    Returns:
        限制后的值
    """
    return max(min_val, min(value, max_val))


def normalize_angle(angle: float) -> float:
    """
    将角度标准化到 [0, 2π) 范围
    
    Args:
        angle: 输入角度 (弧度)
    
    Returns:
        标准化后的角度
    """
    while angle < 0:
        angle += 2 * math.pi
    while angle >= 2 * math.pi:
        angle -= 2 * math.pi
    return angle


def moving_average(values: Sequence[Union[int, float]], window_size: int = 5) -> List[float]:
    """
    计算移动平均
    
    Args:
        values: 数值列表
        window_size: 窗口大小
    
    Returns:
        移动平均列表
    """
    if len(values) < window_size:
        return [float(v) for v in values]
    
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        window_values = values[start_idx:end_idx]
        result.append(sum(float(v) for v in window_values) / len(window_values))
    
    return result
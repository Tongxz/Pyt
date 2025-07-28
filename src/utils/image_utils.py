# Image processing utilities
# 图像处理工具模块

import numpy as np
from typing import Tuple, List, Optional, Union
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def resize_image(image: np.ndarray, 
                target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)
        keep_aspect_ratio: 是否保持宽高比
    
    Returns:
        调整后的图像
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for image resizing")
    
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 调整大小
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建目标尺寸的画布并居中放置
        canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    else:
        return cv2.resize(image, target_size)


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    裁剪图像
    
    Args:
        image: 输入图像
        bbox: 边界框 (x1, y1, x2, y2)
    
    Returns:
        裁剪后的图像
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # 确保坐标在图像范围内
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    return image[y1:y2, x1:x2]


def draw_bbox(image: np.ndarray, 
              bbox: Tuple[int, int, int, int],
              label: str = "",
              color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制边界框
    
    Args:
        image: 输入图像
        bbox: 边界框 (x1, y1, x2, y2)
        label: 标签文本
        color: 颜色 (B, G, R)
        thickness: 线条粗细
    
    Returns:
        绘制后的图像
    """
    if not CV2_AVAILABLE:
        return image
    
    img = image.copy()
    x1, y1, x2, y2 = bbox
    
    # 绘制边界框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # 绘制标签
    if label:
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def draw_keypoints(image: np.ndarray,
                  keypoints: List[Tuple[int, int]],
                  color: Tuple[int, int, int] = (0, 0, 255),
                  radius: int = 3) -> np.ndarray:
    """
    在图像上绘制关键点
    
    Args:
        image: 输入图像
        keypoints: 关键点列表 [(x, y), ...]
        color: 颜色 (B, G, R)
        radius: 点的半径
    
    Returns:
        绘制后的图像
    """
    if not CV2_AVAILABLE:
        return image
    
    img = image.copy()
    for x, y in keypoints:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
    
    return img


def normalize_image(image: np.ndarray, 
                   mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                   std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    标准化图像
    
    Args:
        image: 输入图像 (0-255)
        mean: 均值
        std: 标准差
    
    Returns:
        标准化后的图像
    """
    # 转换为浮点数并归一化到 [0, 1]
    img = image.astype(np.float32) / 255.0
    
    # 标准化
    img = (img - np.array(mean)) / np.array(std)
    
    return img


def denormalize_image(image: np.ndarray,
                     mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    反标准化图像
    
    Args:
        image: 标准化的图像
        mean: 均值
        std: 标准差
    
    Returns:
        反标准化后的图像 (0-255)
    """
    # 反标准化
    img = image * np.array(std) + np.array(mean)
    
    # 转换回 [0, 255] 范围
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    return img


def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    加载图像文件
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        图像数组，如果加载失败返回None
    """
    try:
        if CV2_AVAILABLE:
            image = cv2.imread(str(image_path))
            return image
        elif PIL_AVAILABLE:
            image = Image.open(image_path)
            return np.array(image)
        else:
            raise ImportError("Neither OpenCV nor PIL is available")
    except Exception:
        return None


def save_image(image: np.ndarray, save_path: Union[str, Path]) -> bool:
    """
    保存图像文件
    
    Args:
        image: 图像数组
        save_path: 保存路径
    
    Returns:
        是否保存成功
    """
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if CV2_AVAILABLE:
            return cv2.imwrite(str(save_path), image)
        elif PIL_AVAILABLE:
            Image.fromarray(image).save(save_path)
            return True
        else:
            return False
    except Exception:
        return False


def calculate_iou(bbox1: Tuple[int, int, int, int], 
                 bbox2: Tuple[int, int, int, int]) -> float:
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
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0
#!/usr/bin/env python3
"""
检测参数优化配置
用于解决人体检测数量不准确和发网检测误判问题
"""

# 人体检测优化参数
HUMAN_DETECTION_CONFIG = {
    # 提高置信度阈值，减少误检
    "confidence_threshold": 0.5,  # 从0.3提高到0.5
    # 提高IoU阈值，更严格的重叠抑制
    "iou_threshold": 0.5,  # 从0.4提高到0.5
    # 增加最小检测框面积，过滤小目标
    "min_box_area": 1500,  # 从500提高到1500
    # 降低最大宽高比，过滤异常形状
    "max_box_ratio": 4.0,  # 从8.0降低到4.0
    # 增加最小尺寸要求
    "min_width": 30,  # 最小宽度
    "min_height": 60,  # 最小高度
    # NMS后处理参数
    "nms_threshold": 0.4,  # 非极大值抑制阈值
    "max_detections": 10,  # 最大检测数量限制
}

# 发网检测优化参数
HAIRNET_DETECTION_CONFIG = {
    # 提高边缘密度阈值，减少误判
    "edge_density_threshold": 0.008,  # 从0.002提高到0.008
    # 提高轮廓数量要求
    "min_contour_count": 3,  # 从0提高到3
    # 提高上部区域边缘密度要求
    "upper_edge_density_threshold": 0.012,  # 从0.003提高到0.012
    # 提高浅蓝色比例阈值
    "light_blue_ratio_threshold": 0.002,  # 从0.0003提高到0.002
    # 提高浅色比例阈值
    "light_color_ratio_threshold": 0.005,  # 从0.001提高到0.005
    # 综合得分阈值
    "total_score_threshold": 1.0,  # 从0.2提高到1.0
    # 置信度计算参数
    "base_confidence_weight": 0.6,  # 基础置信度权重
    "color_confidence_weight": 0.4,  # 颜色特征权重
    # 最小置信度要求
    "min_confidence": 0.3,  # 最小置信度阈值
    # 检测策略权重
    "strategy_weights": {
        "light_blue_hairnet": 0.9,
        "general_hairnet": 0.8,
        "light_hairnet": 0.7,
        "basic_hairnet": 0.6,
        "minimal_hairnet": 0.4,  # 降低极宽松检测的权重
    },
}

# 图像预处理参数
IMAGE_PROCESSING_CONFIG = {
    # 图像尺寸标准化
    "target_width": 640,
    "target_height": 640,
    # 头部区域提取参数
    "head_region_ratio": 0.25,  # 头部占人体的比例
    "head_expand_ratio": 0.1,  # 头部区域扩展比例
    # 图像增强参数
    "brightness_adjustment": 1.1,
    "contrast_adjustment": 1.2,
    "saturation_adjustment": 1.1,
}

# 后处理优化参数
POST_PROCESSING_CONFIG = {
    # 结果过滤
    "min_person_confidence": 0.5,  # 人体检测最小置信度
    "min_hairnet_confidence": 0.3,  # 发网检测最小置信度
    # 重复检测过滤
    "duplicate_iou_threshold": 0.7,  # 重复检测IoU阈值
    # 异常结果过滤
    "max_persons_per_image": 8,  # 单张图片最大人数限制
    "min_head_area_ratio": 0.02,  # 头部区域最小面积比例
}


def apply_optimized_config():
    """
    应用优化配置到检测器
    """
    config = {
        "human_detection": HUMAN_DETECTION_CONFIG,
        "hairnet_detection": HAIRNET_DETECTION_CONFIG,
        "image_processing": IMAGE_PROCESSING_CONFIG,
        "post_processing": POST_PROCESSING_CONFIG,
    }

    return config


def get_detection_summary():
    """
    获取检测配置摘要
    """
    return {
        "optimization_target": "提高检测准确性，减少误判",
        "human_detection_changes": ["提高置信度阈值到0.5", "增强NMS参数", "增加最小尺寸要求", "限制最大检测数量"],
        "hairnet_detection_changes": ["提高边缘密度阈值", "增加轮廓数量要求", "提高颜色特征阈值", "降低极宽松检测权重"],
        "expected_improvements": ["减少人体检测的误检数量", "提高发网检测的准确性", "降低假阳性率", "提升整体检测质量"],
    }


if __name__ == "__main__":
    config = apply_optimized_config()
    summary = get_detection_summary()

    print("=== 检测参数优化配置 ===")
    print(f"人体检测置信度阈值: {config['human_detection']['confidence_threshold']}")
    print(f"发网检测边缘密度阈值: {config['hairnet_detection']['edge_density_threshold']}")
    print(f"最大检测人数限制: {config['post_processing']['max_persons_per_image']}")

    print("\n=== 预期改进效果 ===")
    for improvement in summary["expected_improvements"]:
        print(f"- {improvement}")

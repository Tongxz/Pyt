#!/usr/bin/env python3
"""
发网检测参数分析脚本
分析当前检测阈值设置并提供优化建议
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))


def analyze_detection_thresholds():
    """
    分析当前的发网检测阈值设置
    """
    print("=== 发网检测参数分析 ===")
    print()

    print("当前检测阈值设置:")
    print()

    print("1. 浅蓝色发网检测条件:")
    print("   - 浅蓝色比例 > 0.008 (0.8%)")
    print("   - 上部浅蓝色比例 > 0.005 (0.5%)")
    print("   - 边缘密度 > 0.020 (2.0%)")
    print("   - 小轮廓数量 > 4")
    print("   - 上部边缘密度 > 0.025 (2.5%)")
    print()

    print("2. 一般发网检测条件:")
    print("   - 综合得分 > 1.5")
    print("   - 边缘密度 > 0.025 (2.5%)")
    print("   - 小轮廓数量 > 6")
    print("   - 上部边缘密度 > 0.030 (3.0%)")
    print("   - 总颜色比例 > 0.010 (1.0%)")
    print()

    print("3. 浅色发网检测条件:")
    print("   - 颜色特征: 浅蓝色>0.005 或 浅色>0.012 或 白色>0.008 或 绿色>0.005")
    print("   - 边缘密度 > 0.018 (1.8%)")
    print("   - 小轮廓数量 > 4")
    print("   - 上部边缘密度 > 0.022 (2.2%)")
    print("   - 总颜色比例 > 0.008 (0.8%)")
    print()

    print("4. 基础发网检测条件:")
    print("   - 边缘密度 > 0.022 (2.2%)")
    print("   - 小轮廓数量 > 5")
    print("   - 上部边缘密度 > 0.025 (2.5%)")
    print("   - 总颜色比例 > 0.008 (0.8%)")
    print()

    print("5. 最低检测标准 (4个条件任一满足):")
    print("   条件A: 边缘密度>0.020 且 轮廓>4 且 颜色比例>0.007 且 上部边缘>0.025")
    print("   条件B: 颜色比例>0.012 且 上部边缘>0.028 且 轮廓>3 且 边缘密度>0.018")
    print("   条件C: 浅色比例>0.010 且 边缘密度>0.020 且 上部边缘>0.025 且 轮廓>4")
    print("   条件D: 上部边缘>0.035 且 综合得分>2.0 且 轮廓>5 且 颜色比例>0.008")
    print()

    print("=== 问题分析 ===")
    print()
    print("根据用户反馈，P3未佩戴发网但被误检为佩戴，说明:")
    print("1. 当前阈值可能仍然过于宽松")
    print("2. 需要进一步提高检测的严格性")
    print("3. 特别是对于边缘密度和颜色特征的要求")
    print()

    print("=== 优化建议 ===")
    print()
    print("建议调整方向:")
    print("1. 提高边缘密度要求 (当前最低0.018 -> 建议0.025+)")
    print("2. 提高轮廓数量要求 (当前最低3 -> 建议5+)")
    print("3. 提高颜色比例要求 (当前最低0.007 -> 建议0.010+)")
    print("4. 增加多条件组合要求，避免单一特征误检")
    print("5. 特别加强上部区域的检测要求")
    print()

    print("=== 推荐的新阈值设置 ===")
    print()
    print("更严格的检测条件:")
    print("- 最低边缘密度: 0.025 (2.5%)")
    print("- 最低轮廓数量: 5")
    print("- 最低颜色比例: 0.010 (1.0%)")
    print("- 最低上部边缘密度: 0.030 (3.0%)")
    print("- 要求多个条件同时满足，而非单一条件")
    print()

    print("特殊情况处理:")
    print("- 对于明显的发网特征（如强烈的蓝色），可以适当降低其他要求")
    print("- 对于模糊的特征，需要更严格的多重验证")
    print("- 增加头部区域位置验证（发网主要在头顶部分）")
    print()

    return True


def suggest_threshold_adjustment():
    """
    提供具体的阈值调整建议
    """
    print("=== 具体调整建议 ===")
    print()

    adjustments = [
        {
            "name": "保守调整 (减少误检)",
            "changes": {
                "最低边缘密度": "0.018 -> 0.025",
                "最低轮廓数量": "3 -> 5",
                "最低颜色比例": "0.007 -> 0.012",
                "最低上部边缘": "0.022 -> 0.030",
            },
            "expected": "显著减少误检，可能略微增加漏检",
        },
        {
            "name": "平衡调整 (当前推荐)",
            "changes": {
                "边缘密度要求": "适中提升",
                "多条件验证": "增强组合条件",
                "颜色特征": "更精确的阈值",
                "位置验证": "加强头部区域验证",
            },
            "expected": "平衡误检和漏检，整体准确性提升",
        },
        {
            "name": "激进调整 (最小误检)",
            "changes": {
                "最低边缘密度": "0.018 -> 0.030",
                "最低轮廓数量": "3 -> 7",
                "最低颜色比例": "0.007 -> 0.015",
                "综合得分": "1.5 -> 2.5",
            },
            "expected": "最小化误检，但可能增加漏检率",
        },
    ]

    for i, adj in enumerate(adjustments, 1):
        print(f"{i}. {adj['name']}:")
        for key, value in adj["changes"].items():
            print(f"   {key}: {value}")
        print(f"   预期效果: {adj['expected']}")
        print()

    print("推荐使用: 保守调整 -> 测试效果 -> 根据结果进一步微调")
    print()


if __name__ == "__main__":
    analyze_detection_thresholds()
    print()
    suggest_threshold_adjustment()

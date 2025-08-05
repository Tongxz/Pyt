#!/usr/bin/env python3
"""
查看增强版ROI分析结果
"""

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def view_enhanced_results():
    """查看增强版分析结果"""
    result_file = "enhanced_roi_analysis.png"

    if not os.path.exists(result_file):
        print(f"结果文件不存在: {result_file}")
        return

    print(f"显示增强版ROI分析结果: {result_file}")

    # 读取并显示图像
    img = mpimg.imread(result_file)

    plt.figure(figsize=(24, 16))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Enhanced Head ROI Analysis Results", fontsize=18, pad=20)
    plt.tight_layout()
    plt.show()

    print("\n分析结果说明:")
    print("每行显示一个检测到的人体，包含5个分析视图:")
    print("1. 原图 + 人体检测框(红色) + 头部ROI框(蓝色) + 关键点(绿色)")
    print("2. 头部ROI原图 + 提取方法和质量评分")
    print("3. 边缘检测结果 + 边缘密度")
    print("4. 颜色分析热力图 + 蓝色/白色/总体比例")
    print("5. 发网检测结果 + 置信度和颜色信息")
    print("\n改进点:")
    print("- 使用多种方法融合提取头部ROI (关键点、面部检测、改进比例)")
    print("- 自动选择质量最高的ROI提取方法")
    print("- 显示详细的检测信息和质量评分")
    print("- 更准确的头部区域定位")


if __name__ == "__main__":
    view_enhanced_results()

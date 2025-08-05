#!/usr/bin/env python3
"""
查看ROI可视化分析结果
"""

import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def view_roi_analysis(image_path: str):
    """查看ROI分析结果"""
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return

    try:
        # 读取图像
        img = mpimg.imread(image_path)

        # 创建图形
        plt.figure(figsize=(20, 12))
        plt.imshow(img)
        plt.title(
            "ROI可视化分析结果\n每行显示：原图+人体框+头部ROI | 头部ROI原图 | 边缘检测 | 发网检测结果",
            fontsize=14,
            pad=20,
        )
        plt.axis("off")

        # 添加说明文字
        plt.figtext(
            0.02,
            0.02,
            "说明：\n"
            "• 红色框：人体检测边界框\n"
            "• 蓝色框：头部ROI区域\n"
            "• 绿色点：关键点检测\n"
            "• 边缘检测：显示网状结构特征\n"
            "• 发网检测：热力图显示颜色特征",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

        print(f"ROI分析结果显示完成: {image_path}")

    except Exception as e:
        print(f"显示图像失败: {e}")


def main():
    """主函数"""
    # 默认查看test_person的ROI分析结果
    default_file = "test_person_roi_analysis.png"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_file

    print(f"查看ROI分析结果: {image_path}")
    view_roi_analysis(image_path)


if __name__ == "__main__":
    main()

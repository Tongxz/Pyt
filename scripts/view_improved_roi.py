#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看改进的头部ROI对比结果
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

def view_roi_comparison_results():
    """查看ROI对比结果"""
    # 查找所有对比结果文件
    pattern = "improved_head_roi_comparison_person_*.png"
    result_files = glob.glob(pattern)
    
    if not result_files:
        print("未找到对比结果文件")
        return
    
    result_files.sort()  # 按文件名排序
    
    print(f"找到 {len(result_files)} 个对比结果文件:")
    for i, file in enumerate(result_files, 1):
        print(f"{i}. {file}")
    
    # 显示每个结果
    for file in result_files:
        if os.path.exists(file):
            print(f"\n显示: {file}")
            
            # 读取并显示图像
            img = mpimg.imread(file)
            
            plt.figure(figsize=(20, 14))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Head ROI Extraction Comparison - {file}', fontsize=16, pad=20)
            plt.tight_layout()
            plt.show()
        else:
            print(f"文件不存在: {file}")

if __name__ == '__main__':
    view_roi_comparison_results()
#!/usr/bin/env python3
"""
增强版ROI可视化工具
集成改进的头部检测算法
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improved_head_roi import ImprovedHeadROIExtractor

from src.core.detector import HumanDetector
from src.core.hairnet_detector import HairnetDetector


class EnhancedROIVisualizer:
    """增强版ROI可视化器"""

    def __init__(self):
        """初始化可视化器"""
        self.human_detector = HumanDetector()
        self.hairnet_detector = HairnetDetector()
        self.head_extractor = ImprovedHeadROIExtractor()

    def analyze_and_visualize_enhanced(self, image: np.ndarray) -> Optional[Figure]:
        """增强版分析和可视化"""
        # 检测人体
        detections = self.human_detector.detect(image)

        if not detections:
            print("未检测到人体")
            return None

        print(f"检测到 {len(detections)} 个人体")

        # 为每个人体创建分析结果
        num_persons = len(detections)
        fig = plt.figure(figsize=(20, 6 * num_persons))
        gs = GridSpec(num_persons, 5, figure=fig, hspace=0.3, wspace=0.2)

        for person_idx, detection in enumerate(detections):
            bbox = detection["bbox"]
            keypoints = detection.get("keypoints", None)
            confidence = detection.get("confidence", 0.0)

            print(f"\n处理第 {person_idx + 1} 个人体:")
            print(f"  边界框: {bbox}")
            print(f"  置信度: {confidence:.3f}")
            print(f"  关键点数量: {len(keypoints) // 3 if keypoints else 0}")

            # 使用改进的多方法头部ROI提取
            head_roi, roi_info = self.head_extractor.extract_head_roi_multi_method(
                image, bbox, keypoints
            )

            if head_roi is None or head_roi.size == 0 or roi_info is None:
                print(f"  警告: 无法提取第 {person_idx + 1} 个人体的头部ROI")
                continue

            print(f"  头部ROI提取成功:")
            print(f"    方法: {roi_info.get('method', 'unknown')}")
            print(f"    尺寸: {roi_info.get('size', (0, 0))}")
            print(f"    质量评分: {roi_info.get('best_score', 0):.3f}")

            # 进行发网检测
            hairnet_result = self.hairnet_detector._detect_hairnet_with_pytorch(
                head_roi
            )
            wearing_hairnet = hairnet_result.get("wearing_hairnet", False)
            hairnet_confidence = hairnet_result.get("confidence", 0.0)
            hairnet_color = hairnet_result.get("hairnet_color", "unknown")

            print(f"  发网检测结果:")
            print(f"    佩戴发网: {wearing_hairnet}")
            print(f"    置信度: {hairnet_confidence:.3f}")
            print(f"    颜色: {hairnet_color}")

            # 可视化结果
            row = person_idx

            # 1. 原图 + 人体检测框 + 头部ROI框
            ax1 = fig.add_subplot(gs[row, 0])
            ax1.imshow(image)
            ax1.set_title(
                f"Person {person_idx + 1}\n人体检测 (conf: {confidence:.2f})", fontsize=10
            )

            # 绘制人体边界框
            x1, y1, x2, y2 = bbox
            person_rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax1.add_patch(person_rect)

            # 绘制头部ROI框
            if roi_info and "bbox" in roi_info:
                hx1, hy1, hx2, hy2 = roi_info["bbox"]
                head_rect = patches.Rectangle(
                    (hx1, hy1),
                    hx2 - hx1,
                    hy2 - hy1,
                    linewidth=2,
                    edgecolor="blue",
                    facecolor="none",
                )
                ax1.add_patch(head_rect)

            # 绘制关键点
            if keypoints and len(keypoints) >= 15:
                for i in range(0, min(15, len(keypoints)), 3):
                    x, y, conf = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                    if x > 0 and y > 0 and conf > 0.3:
                        ax1.plot(x, y, "go", markersize=4)

            ax1.axis("off")

            # 2. 头部ROI原图
            ax2 = fig.add_subplot(gs[row, 1])
            ax2.imshow(head_roi)
            method_info = roi_info.get("method", "unknown") if roi_info else "unknown"
            quality_score = roi_info.get("best_score", 0) if roi_info else 0
            ax2.set_title(
                f"头部ROI\n方法: {method_info}\n质量: {quality_score:.2f}", fontsize=10
            )
            ax2.axis("off")

            # 3. 边缘检测
            ax3 = fig.add_subplot(gs[row, 2])
            if len(head_roi.shape) == 3:
                gray_roi = cv2.cvtColor(head_roi, cv2.COLOR_RGB2GRAY)
            else:
                gray_roi = head_roi

            edges = cv2.Canny(gray_roi, 50, 150)
            ax3.imshow(edges, cmap="gray")

            # 计算边缘密度
            edge_density = np.sum(edges > 0) / edges.size
            ax3.set_title(f"边缘检测\n密度: {edge_density:.3f}", fontsize=10)
            ax3.axis("off")

            # 4. 颜色分析
            ax4 = fig.add_subplot(gs[row, 3])

            # HSV颜色空间分析
            hsv = cv2.cvtColor(head_roi, cv2.COLOR_RGB2HSV)

            # 浅蓝色发网检测
            light_blue_lower = np.array([90, 20, 80])
            light_blue_upper = np.array([140, 255, 255])
            light_blue_mask = cv2.inRange(hsv, light_blue_lower, light_blue_upper)

            # 白色发网检测
            white_lower = np.array([0, 0, 180])
            white_upper = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, white_lower, white_upper)

            # 组合掩码
            combined_mask = cv2.bitwise_or(light_blue_mask, white_mask)

            ax4.imshow(combined_mask, cmap="hot")

            # 计算颜色比例
            blue_ratio = np.sum(light_blue_mask > 0) / light_blue_mask.size
            white_ratio = np.sum(white_mask > 0) / white_mask.size
            total_ratio = np.sum(combined_mask > 0) / combined_mask.size

            ax4.set_title(
                f"颜色分析\n蓝色: {blue_ratio:.3f}\n白色: {white_ratio:.3f}\n总计: {total_ratio:.3f}",
                fontsize=10,
            )
            ax4.axis("off")

            # 5. 发网检测结果
            ax5 = fig.add_subplot(gs[row, 4])

            # 创建结果可视化
            result_img = head_roi.copy()

            # 在图像上叠加检测结果
            if wearing_hairnet:
                # 绘制发网区域
                colored_mask = np.zeros_like(head_roi)
                colored_mask[combined_mask > 0] = [0, 255, 0]  # 绿色表示检测到发网
                result_img = cv2.addWeighted(result_img, 0.7, colored_mask, 0.3, 0)

            ax5.imshow(result_img)

            # 显示检测结果文本
            status = "佩戴发网" if wearing_hairnet else "未佩戴发网"
            color = "green" if wearing_hairnet else "red"

            ax5.set_title(
                f"发网检测结果\n{status}\n置信度: {hairnet_confidence:.3f}\n颜色: {hairnet_color}",
                fontsize=10,
                color=color,
            )
            ax5.axis("off")

        # 添加总体标题
        fig.suptitle(
            "Enhanced Hairnet Detection ROI Analysis Results", fontsize=16, y=0.98
        )

        return fig

    def process_image(self, image_path: str, save_path: Optional[str] = None) -> bool:
        """处理单张图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 分析和可视化
        result_fig = self.analyze_and_visualize_enhanced(image_rgb)

        if result_fig is not None:
            if save_path:
                # 保存结果
                result_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"\n增强版分析结果已保存: {save_path}")
                plt.close(result_fig)
            else:
                # 显示结果
                plt.show()
                plt.close(result_fig)
            return True
        else:
            print("分析失败")
            return False

    def batch_process(self, image_dir: str, output_dir: str) -> None:
        """批量处理图像"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        processed_count = 0

        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(image_dir, filename)
                output_path = os.path.join(output_dir, f"enhanced_{filename}")

                print(f"\n处理: {filename}")
                if self.process_image(image_path, output_path):
                    processed_count += 1

        print(f"\n批量处理完成，共处理 {processed_count} 张图像")


def main():
    parser = argparse.ArgumentParser(description="增强版ROI可视化工具")
    parser.add_argument("--image", "-i", type=str, help="单张图像路径")
    parser.add_argument("--batch", "-b", type=str, help="批量处理的图像目录")
    parser.add_argument("--output", "-o", type=str, help="输出路径或目录")
    parser.add_argument("--save", "-s", action="store_true", help="保存结果而不是显示")

    args = parser.parse_args()

    visualizer = EnhancedROIVisualizer()

    if args.image:
        # 单张图像处理
        save_path = args.output if args.save else None
        visualizer.process_image(args.image, save_path)
    elif args.batch:
        # 批量处理
        output_dir = args.output or "enhanced_roi_results"
        visualizer.batch_process(args.batch, output_dir)
    else:
        print("请指定 --image 或 --batch 参数")
        parser.print_help()


if __name__ == "__main__":
    main()

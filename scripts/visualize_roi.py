#!/usr/bin/env python3
"""
发网检测ROI可视化工具
用于检查头部ROI提取是否准确对准发网区域
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.detector import HumanDetector
from src.core.hairnet_detector import HairnetDetector


class ROIVisualizer:
    """ROI可视化器"""

    def __init__(self):
        """初始化可视化器"""
        self.human_detector = HumanDetector()
        self.hairnet_detector = HairnetDetector()

    def extract_head_roi(
        self,
        image: np.ndarray,
        bbox: List[float],
        keypoints: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """提取头部ROI区域"""
        x1, y1, x2, y2 = map(int, bbox)

        # 计算头部区域（上半身的上1/3部分）
        person_height = y2 - y1
        person_width = x2 - x1

        # 头部区域估算
        head_height = int(person_height * 0.25)  # 头部约占身高的1/4
        head_width = int(person_width * 0.8)  # 头部宽度约为肩宽的80%

        # 头部中心位置
        head_center_x = (x1 + x2) // 2
        head_top = y1 + int(person_height * 0.05)  # 稍微向下偏移避免截断

        # 计算头部边界框
        head_x1 = max(0, head_center_x - head_width // 2)
        head_y1 = max(0, head_top)
        head_x2 = min(image.shape[1], head_center_x + head_width // 2)
        head_y2 = min(image.shape[0], head_top + head_height)

        # 如果有关键点信息，使用关键点优化头部区域
        if keypoints and len(keypoints) >= 15:  # YOLO pose有17个关键点
            # 关键点索引：0-鼻子, 1-左眼, 2-右眼, 3-左耳, 4-右耳
            nose_x, nose_y = keypoints[0], keypoints[1]
            left_eye_x, left_eye_y = keypoints[3], keypoints[4]
            right_eye_x, right_eye_y = keypoints[6], keypoints[7]
            left_ear_x, left_ear_y = keypoints[9], keypoints[10]
            right_ear_x, right_ear_y = keypoints[12], keypoints[13]

            # 如果关键点可见，使用关键点信息
            visible_points = []
            if nose_x > 0 and nose_y > 0:
                visible_points.append((nose_x, nose_y))
            if left_eye_x > 0 and left_eye_y > 0:
                visible_points.append((left_eye_x, left_eye_y))
            if right_eye_x > 0 and right_eye_y > 0:
                visible_points.append((right_eye_x, right_eye_y))
            if left_ear_x > 0 and left_ear_y > 0:
                visible_points.append((left_ear_x, left_ear_y))
            if right_ear_x > 0 and right_ear_y > 0:
                visible_points.append((right_ear_x, right_ear_y))

            if len(visible_points) >= 2:
                # 使用关键点重新计算头部区域
                points_array = np.array(visible_points)
                min_x, min_y = np.min(points_array, axis=0)
                max_x, max_y = np.max(points_array, axis=0)

                # 扩展边界以包含整个头部
                margin_x = int((max_x - min_x) * 0.3)
                margin_y = int((max_y - min_y) * 0.4)

                head_x1 = max(0, int(min_x - margin_x))
                head_y1 = max(0, int(min_y - margin_y))
                head_x2 = min(image.shape[1], int(max_x + margin_x))
                head_y2 = min(image.shape[0], int(max_y + margin_y * 1.5))  # 下方多留一些空间

        # 提取ROI
        roi = image[head_y1:head_y2, head_x1:head_x2]

        roi_info = {
            "bbox": [head_x1, head_y1, head_x2, head_y2],
            "size": (head_x2 - head_x1, head_y2 - head_y1),
            "method": "keypoints"
            if keypoints and len(keypoints) >= 15
            else "bbox_estimation",
        }

        return roi, roi_info

    def visualize_single_image(self, image_path: str, save_path: Optional[str] = None):
        """可视化单张图像的ROI提取结果"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 检测人体
        detections = self.human_detector.detect(image)

        if not detections:
            print("未检测到人体")
            return

        # 创建可视化布局
        num_persons = len(detections)
        fig = plt.figure(figsize=(20, 6 * num_persons))
        gs = GridSpec(num_persons, 4, figure=fig, hspace=0.3, wspace=0.3)

        for i, detection in enumerate(detections):
            bbox = detection["bbox"]
            keypoints = detection.get("keypoints", None)
            confidence = detection.get("confidence", 0.0)

            # 提取头部ROI
            roi, roi_info = self.extract_head_roi(image_rgb, bbox, keypoints)

            # 进行发网检测
            hairnet_result = self.hairnet_detector._detect_hairnet_with_pytorch(roi)

            # 1. 原始图像 + 人体框
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(image_rgb)
            ax1.set_title(f"Person {i+1}\nConfidence: {confidence:.2f}")

            # 绘制人体边界框
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax1.add_patch(rect)

            # 绘制头部ROI框
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

            # 绘制关键点（如果有）
            if keypoints and len(keypoints) >= 15:
                for j in range(0, min(15, len(keypoints)), 3):
                    x, y, conf = keypoints[j], keypoints[j + 1], keypoints[j + 2]
                    if x > 0 and y > 0 and conf > 0.5:
                        ax1.plot(x, y, "go", markersize=4)

            ax1.set_xlim(0, image_rgb.shape[1])
            ax1.set_ylim(image_rgb.shape[0], 0)
            ax1.axis("off")

            # 2. 头部ROI原图
            ax2 = fig.add_subplot(gs[i, 1])
            if roi.size > 0:
                ax2.imshow(roi)
                ax2.set_title(
                    f'Head ROI\nSize: {roi_info["size"]}\nMethod: {roi_info["method"]}'
                )
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "ROI提取失败",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
                ax2.set_title("Head ROI (Failed)")
            ax2.axis("off")

            # 3. 边缘检测结果
            ax3 = fig.add_subplot(gs[i, 2])
            if roi.size > 0:
                # 转换为灰度图
                roi_gray = (
                    cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                    if len(roi.shape) == 3
                    else roi
                )
                edges = cv2.Canny(roi_gray, 50, 150)
                ax3.imshow(edges, cmap="gray")
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                ax3.set_title(f"Edge Detection\nDensity: {edge_density:.4f}")
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "无ROI数据",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                )
                ax3.set_title("Edge Detection (N/A)")
            ax3.axis("off")

            # 4. 发网检测结果
            ax4 = fig.add_subplot(gs[i, 3])
            if roi.size > 0:
                # 显示颜色掩码
                hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

                # 浅蓝色掩码
                light_blue_lower = np.array([90, 20, 80])
                light_blue_upper = np.array([140, 255, 255])
                light_blue_mask = cv2.inRange(hsv, light_blue_lower, light_blue_upper)

                # 浅色掩码
                light_colors_lower = np.array([0, 0, 120])
                light_colors_upper = np.array([180, 120, 255])
                light_mask = cv2.inRange(hsv, light_colors_lower, light_colors_upper)

                # 组合掩码
                combined_mask = cv2.bitwise_or(light_blue_mask, light_mask)
                ax4.imshow(combined_mask, cmap="hot")

                # 显示检测结果
                wearing_hairnet = hairnet_result.get("wearing_hairnet", False)
                confidence = hairnet_result.get("confidence", 0.0)
                color = hairnet_result.get("hairnet_color", "unknown")

                title = f"Hairnet Detection\n"
                title += f'Result: {"YES" if wearing_hairnet else "NO"}\n'
                title += f"Confidence: {confidence:.3f}\n"
                title += f"Color: {color}"
                ax4.set_title(title)
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "无ROI数据",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                )
                ax4.set_title("Hairnet Detection (N/A)")
            ax4.axis("off")

        plt.suptitle(f"ROI Visualization: {os.path.basename(image_path)}", fontsize=16)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"可视化结果已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_batch(self, image_dir: str, output_dir: str = "roi_visualizations"):
        """批量可视化图像目录中的所有图像"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []

        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_dir, file))

        if not image_files:
            print(f"在目录 {image_dir} 中未找到图像文件")
            return

        print(f"找到 {len(image_files)} 张图像，开始批量处理...")

        for i, image_path in enumerate(image_files):
            print(f"处理第 {i+1}/{len(image_files)} 张图像: {os.path.basename(image_path)}")

            output_name = (
                os.path.splitext(os.path.basename(image_path))[0] + "_roi_analysis.png"
            )
            output_path = os.path.join(output_dir, output_name)

            try:
                self.visualize_single_image(image_path, output_path)
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {e}")

        print(f"批量处理完成，结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="发网检测ROI可视化工具")
    parser.add_argument("--image", "-i", type=str, help="单张图像路径")
    parser.add_argument("--dir", "-d", type=str, help="图像目录路径")
    parser.add_argument(
        "--output", "-o", type=str, default="roi_visualizations", help="输出目录"
    )
    parser.add_argument("--save", "-s", type=str, help="保存单张图像结果的路径")

    args = parser.parse_args()

    visualizer = ROIVisualizer()

    if args.image:
        # 单张图像处理
        visualizer.visualize_single_image(args.image, args.save)
    elif args.dir:
        # 批量处理
        visualizer.visualize_batch(args.dir, args.output)
    else:
        print("请指定 --image 或 --dir 参数")
        print("使用示例:")
        print("  python visualize_roi.py --image test_image.jpg")
        print("  python visualize_roi.py --dir ./test_images --output ./results")


if __name__ == "__main__":
    main()

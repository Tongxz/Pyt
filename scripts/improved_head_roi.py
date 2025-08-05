#!/usr/bin/env python3
"""
改进的头部ROI提取算法
结合多种方法提高头部区域定位的准确性
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.core.detector import HumanDetector

# 避免循环导入，不直接导入HairnetDetector


class ImprovedHeadROIExtractor:
    """改进的头部ROI提取器"""

    def __init__(self):
        """初始化提取器"""
        self.human_detector = HumanDetector()
        # 移除对HairnetDetector的依赖以避免循环导入

        # 初始化面部检测器
        try:
            import cv2

            # 尝试多种路径来找到Haar级联分类器
            possible_paths = []

            # 尝试使用cv2.data.haarcascades（如果可用）
            try:
                if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                    possible_paths.append(cv2.data.haarcascades)
            except:
                pass

            # 添加其他可能的路径
            possible_paths.extend(
                [
                    os.path.join(cv2.__path__[0], "data", "haarcascades"),
                    cv2.__file__.replace("__init__.py", "data/haarcascades/"),
                    "/usr/share/opencv4/haarcascades/",  # Linux系统路径
                    "/opt/homebrew/share/opencv4/haarcascades/",  # macOS Homebrew路径
                ]
            )

            self.face_cascade = None
            self.profile_cascade = None

            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        front_face_path = os.path.join(
                            path, "haarcascade_frontalface_default.xml"
                        )
                        profile_face_path = os.path.join(
                            path, "haarcascade_profileface.xml"
                        )

                        if os.path.exists(front_face_path):
                            self.face_cascade = cv2.CascadeClassifier(front_face_path)
                            if not self.face_cascade.empty():
                                print(f"成功加载正面人脸检测器: {front_face_path}")
                                break
                except:
                    continue

            # 尝试加载侧面人脸检测器（可选）
            if self.face_cascade and not self.face_cascade.empty():
                for path in possible_paths:
                    try:
                        if os.path.exists(path):
                            profile_face_path = os.path.join(
                                path, "haarcascade_profileface.xml"
                            )
                            if os.path.exists(profile_face_path):
                                self.profile_cascade = cv2.CascadeClassifier(
                                    profile_face_path
                                )
                                if not self.profile_cascade.empty():
                                    print(f"成功加载侧面人脸检测器: {profile_face_path}")
                                    break
                    except:
                        continue

            self.face_detection_available = (
                self.face_cascade is not None and not self.face_cascade.empty()
            )

            if not self.face_detection_available:
                print("警告: 无法找到Haar级联分类器文件，面部检测功能不可用")
                print("尝试的路径:", possible_paths)

        except Exception as e:
            print(f"警告: 面部检测器初始化失败: {e}，将使用备用方法")
            self.face_detection_available = False

    def detect_faces_in_region(
        self, image: np.ndarray, region_bbox: List[int]
    ) -> List[Dict]:
        """在指定区域内检测面部"""
        if not self.face_detection_available or self.face_cascade is None:
            return []

        x1, y1, x2, y2 = region_bbox
        roi_region = image[y1:y2, x1:x2]
        gray_roi = (
            cv2.cvtColor(roi_region, cv2.COLOR_RGB2GRAY)
            if len(roi_region.shape) == 3
            else roi_region
        )

        faces = []

        # 检测正面人脸
        frontal_faces = self.face_cascade.detectMultiScale(
            gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for fx, fy, fw, fh in frontal_faces:
            faces.append(
                {
                    "bbox": [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh],
                    "confidence": 0.8,  # 假设置信度
                    "type": "frontal",
                }
            )

        # 检测侧面人脸（如果可用）
        if self.profile_cascade is not None and not self.profile_cascade.empty():
            profile_faces = self.profile_cascade.detectMultiScale(
                gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for fx, fy, fw, fh in profile_faces:
                faces.append(
                    {
                        "bbox": [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh],
                        "confidence": 0.7,  # 侧面检测置信度稍低
                        "type": "profile",
                    }
                )

        return faces

    def extract_head_roi_with_keypoints(
        self,
        image: np.ndarray,
        bbox: List[float],
        keypoints: Optional[List[float]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """使用关键点提取头部ROI"""
        if not keypoints or len(keypoints) < 15:
            return None, None

        # YOLO pose关键点索引 (x, y, confidence)
        keypoint_indices = {
            "nose": (0, 1, 2),
            "left_eye": (3, 4, 5),
            "right_eye": (6, 7, 8),
            "left_ear": (9, 10, 11),
            "right_ear": (12, 13, 14),
        }

        visible_points = []
        head_keypoints = {}

        for name, (x_idx, y_idx, conf_idx) in keypoint_indices.items():
            if (
                x_idx < len(keypoints)
                and y_idx < len(keypoints)
                and conf_idx < len(keypoints)
            ):
                x, y, conf = keypoints[x_idx], keypoints[y_idx], keypoints[conf_idx]
                if x > 0 and y > 0 and conf > 0.3:  # 置信度阈值
                    visible_points.append((x, y))
                    head_keypoints[name] = (x, y, conf)

        if len(visible_points) < 2:
            return None, None

        # 计算头部边界框
        points_array = np.array(visible_points)
        min_x, min_y = np.min(points_array, axis=0)
        max_x, max_y = np.max(points_array, axis=0)

        # 智能扩展边界
        head_width = max_x - min_x
        head_height = max_y - min_y

        # 根据检测到的关键点类型调整扩展策略
        if "nose" in head_keypoints and (
            "left_eye" in head_keypoints or "right_eye" in head_keypoints
        ):
            # 有鼻子和眼睛，可以更精确地定位
            margin_x = max(head_width * 0.4, 20)  # 至少20像素边距
            margin_y_top = max(head_height * 0.3, 15)  # 上方边距
            margin_y_bottom = max(head_height * 0.6, 25)  # 下方边距（包含下巴和发网区域）
        else:
            # 只有耳朵等关键点，需要更大的边距
            margin_x = max(head_width * 0.5, 25)
            margin_y_top = max(head_height * 0.4, 20)
            margin_y_bottom = max(head_height * 0.8, 30)

        # 计算最终边界框
        head_x1 = max(0, int(min_x - margin_x))
        head_y1 = max(0, int(min_y - margin_y_top))
        head_x2 = min(image.shape[1], int(max_x + margin_x))
        head_y2 = min(image.shape[0], int(max_y + margin_y_bottom))

        roi = image[head_y1:head_y2, head_x1:head_x2]

        roi_info = {
            "bbox": [head_x1, head_y1, head_x2, head_y2],
            "size": (head_x2 - head_x1, head_y2 - head_y1),
            "method": "keypoints_enhanced",
            "keypoints_used": list(head_keypoints.keys()),
            "keypoints_count": len(visible_points),
        }

        return roi, roi_info

    def extract_head_roi_with_face_detection(
        self, image: np.ndarray, bbox: List[float]
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """使用面部检测提取头部ROI"""
        x1, y1, x2, y2 = map(int, bbox)

        # 在人体上半部分搜索面部
        search_height = int((y2 - y1) * 0.6)  # 搜索上60%区域
        search_region = [x1, y1, x2, y1 + search_height]

        faces = self.detect_faces_in_region(image, search_region)

        if not faces:
            return None, None

        # 选择最大的面部（通常是最可靠的）
        best_face = max(
            faces,
            key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
        )

        fx1, fy1, fx2, fy2 = best_face["bbox"]
        face_width = fx2 - fx1
        face_height = fy2 - fy1

        # 扩展面部区域以包含完整头部和发网区域
        margin_x = int(face_width * 0.3)  # 左右扩展30%
        margin_y_top = int(face_height * 0.4)  # 上方扩展40%（包含额头和发网）
        margin_y_bottom = int(face_height * 0.2)  # 下方扩展20%（包含下巴）

        head_x1 = max(0, fx1 - margin_x)
        head_y1 = max(0, fy1 - margin_y_top)
        head_x2 = min(image.shape[1], fx2 + margin_x)
        head_y2 = min(image.shape[0], fy2 + margin_y_bottom)

        roi = image[head_y1:head_y2, head_x1:head_x2]

        roi_info = {
            "bbox": [head_x1, head_y1, head_x2, head_y2],
            "size": (head_x2 - head_x1, head_y2 - head_y1),
            "method": "face_detection",
            "face_bbox": best_face["bbox"],
            "face_type": best_face["type"],
            "face_confidence": best_face["confidence"],
        }

        return roi, roi_info

    def extract_head_roi_proportional(
        self, image: np.ndarray, bbox: List[float]
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """基于比例的头部ROI提取（改进版）"""
        x1, y1, x2, y2 = map(int, bbox)

        person_height = y2 - y1
        person_width = x2 - x1

        # 改进的头部区域估算
        # 根据人体姿态调整头部比例
        aspect_ratio = person_width / person_height

        if aspect_ratio > 0.6:  # 较宽的人体，可能是侧身或手臂张开
            head_height = int(person_height * 0.22)
            head_width = int(person_width * 0.7)
        else:  # 正常比例
            head_height = int(person_height * 0.25)
            head_width = int(person_width * 0.8)

        # 头部位置优化
        head_center_x = (x1 + x2) // 2
        # 头部通常在人体顶部，但要考虑可能的遮挡
        head_top = y1 + int(person_height * 0.02)  # 从顶部2%开始

        # 计算头部边界框
        head_x1 = max(0, head_center_x - head_width // 2)
        head_y1 = max(0, head_top)
        head_x2 = min(image.shape[1], head_center_x + head_width // 2)
        head_y2 = min(image.shape[0], head_top + head_height)

        roi = image[head_y1:head_y2, head_x1:head_x2]

        roi_info = {
            "bbox": [head_x1, head_y1, head_x2, head_y2],
            "size": (head_x2 - head_x1, head_y2 - head_y1),
            "method": "proportional_improved",
            "aspect_ratio": aspect_ratio,
        }

        return roi, roi_info

    def extract_head_roi_multi_method(
        self,
        image: np.ndarray,
        bbox: List[float],
        keypoints: Optional[List[float]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """多方法融合的头部ROI提取"""
        methods_tried = []
        best_roi = None
        best_info = None
        best_score = 0

        # 方法1: 关键点检测
        if keypoints:
            roi_kp, info_kp = self.extract_head_roi_with_keypoints(
                image, bbox, keypoints
            )
            if roi_kp is not None and info_kp is not None and roi_kp.size > 0:
                score = self.evaluate_roi_quality(roi_kp, info_kp)
                methods_tried.append(("keypoints", score))
                if score > best_score:
                    best_roi, best_info, best_score = roi_kp, info_kp, score

        # 方法2: 面部检测
        if self.face_detection_available:
            roi_face, info_face = self.extract_head_roi_with_face_detection(image, bbox)
            if roi_face is not None and info_face is not None and roi_face.size > 0:
                score = self.evaluate_roi_quality(roi_face, info_face)
                methods_tried.append(("face_detection", score))
                if score > best_score:
                    best_roi, best_info, best_score = roi_face, info_face, score

        # 方法3: 改进的比例方法（备用）
        roi_prop, info_prop = self.extract_head_roi_proportional(image, bbox)
        if roi_prop is not None and info_prop is not None and roi_prop.size > 0:
            score = self.evaluate_roi_quality(roi_prop, info_prop)
            methods_tried.append(("proportional", score))
            if score > best_score:
                best_roi, best_info, best_score = roi_prop, info_prop, score

        # 添加方法尝试信息
        if best_info:
            best_info["methods_tried"] = methods_tried
            best_info["best_score"] = best_score

        return best_roi, best_info

    def evaluate_roi_quality(self, roi: np.ndarray, roi_info: Dict) -> float:
        """评估ROI质量"""
        if roi is None or roi.size == 0:
            return 0.0

        score = 0.0

        # 尺寸评分（合理的头部尺寸）
        width, height = roi_info["size"]
        if 40 <= width <= 200 and 50 <= height <= 250:
            score += 0.3
        elif 30 <= width <= 300 and 40 <= height <= 350:
            score += 0.2

        # 宽高比评分（头部通常接近正方形或略高）
        aspect_ratio = width / height if height > 0 else 0
        if 0.7 <= aspect_ratio <= 1.3:
            score += 0.2
        elif 0.5 <= aspect_ratio <= 1.8:
            score += 0.1

        # 方法特定加分
        method = roi_info.get("method", "")
        if method == "keypoints_enhanced":
            keypoints_count = roi_info.get("keypoints_count", 0)
            score += min(0.3, keypoints_count * 0.1)  # 关键点越多越好
        elif method == "face_detection":
            face_confidence = roi_info.get("face_confidence", 0)
            score += face_confidence * 0.2

        # 图像质量评分（简单的方差检查）
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray_roi = roi

        variance = np.var(gray_roi)
        if variance > 100:  # 有足够的细节
            score += 0.2
        elif variance > 50:
            score += 0.1

        return min(1.0, score)  # 限制在1.0以内

    def visualize_comparison(self, image_path: str, save_path: Optional[str] = None):
        """可视化不同方法的头部ROI提取结果对比"""
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

        # 为每个检测到的人体创建对比图
        for person_idx, detection in enumerate(detections):
            bbox = detection["bbox"]
            keypoints = detection.get("keypoints", None)

            # 尝试不同的方法
            roi_kp, info_kp = self.extract_head_roi_with_keypoints(
                image_rgb, bbox, keypoints
            )
            roi_face, info_face = self.extract_head_roi_with_face_detection(
                image_rgb, bbox
            )
            roi_prop, info_prop = self.extract_head_roi_proportional(image_rgb, bbox)
            roi_multi, info_multi = self.extract_head_roi_multi_method(
                image_rgb, bbox, keypoints
            )

            # 创建对比图
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f"头部ROI提取方法对比 - Person {person_idx + 1}", fontsize=16)

            # 原图
            axes[0, 0].imshow(image_rgb)
            axes[0, 0].set_title("原图 + 人体检测框")
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            axes[0, 0].add_patch(rect)
            axes[0, 0].axis("off")

            # 关键点方法
            if roi_kp is not None and roi_kp.size > 0:
                axes[0, 1].imshow(roi_kp)
                score = self.evaluate_roi_quality(roi_kp, info_kp)
                axes[0, 1].set_title(
                    f'关键点方法\n质量评分: {score:.2f}\n关键点数: {info_kp.get("keypoints_count", 0)}'
                )
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "关键点方法失败",
                    ha="center",
                    va="center",
                    transform=axes[0, 1].transAxes,
                )
                axes[0, 1].set_title("关键点方法 (失败)")
            axes[0, 1].axis("off")

            # 面部检测方法
            if roi_face is not None and roi_face.size > 0:
                axes[0, 2].imshow(roi_face)
                score = self.evaluate_roi_quality(roi_face, info_face)
                axes[0, 2].set_title(
                    f'面部检测方法\n质量评分: {score:.2f}\n类型: {info_face.get("face_type", "unknown")}'
                )
            else:
                axes[0, 2].text(
                    0.5,
                    0.5,
                    "面部检测失败",
                    ha="center",
                    va="center",
                    transform=axes[0, 2].transAxes,
                )
                axes[0, 2].set_title("面部检测方法 (失败)")
            axes[0, 2].axis("off")

            # 比例方法
            if roi_prop is not None and roi_prop.size > 0:
                axes[1, 0].imshow(roi_prop)
                score = self.evaluate_roi_quality(roi_prop, info_prop)
                axes[1, 0].set_title(
                    f'改进比例方法\n质量评分: {score:.2f}\n宽高比: {info_prop.get("aspect_ratio", 0):.2f}'
                )
            else:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "比例方法失败",
                    ha="center",
                    va="center",
                    transform=axes[1, 0].transAxes,
                )
                axes[1, 0].set_title("比例方法 (失败)")
            axes[1, 0].axis("off")

            # 多方法融合
            if roi_multi is not None and roi_multi.size > 0:
                axes[1, 1].imshow(roi_multi)
                score = self.evaluate_roi_quality(roi_multi, info_multi)
                best_method = info_multi.get("method", "unknown")
                axes[1, 1].set_title(
                    f"多方法融合 (最佳)\n质量评分: {score:.2f}\n选择方法: {best_method}"
                )
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "多方法融合失败",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("多方法融合 (失败)")
            axes[1, 1].axis("off")

            # 发网检测结果（简化版本，避免循环导入）
            if roi_multi is not None and roi_multi.size > 0:
                # 显示发网检测的颜色掩码
                hsv = cv2.cvtColor(roi_multi, cv2.COLOR_RGB2HSV)
                light_blue_lower = np.array([90, 20, 80])
                light_blue_upper = np.array([140, 255, 255])
                light_blue_mask = cv2.inRange(hsv, light_blue_lower, light_blue_upper)

                # 简单的发网检测逻辑
                mask_ratio = np.sum(light_blue_mask > 0) / (
                    light_blue_mask.shape[0] * light_blue_mask.shape[1]
                )
                wearing_hairnet = mask_ratio > 0.05  # 如果蓝色区域超过5%

                axes[1, 2].imshow(light_blue_mask, cmap="hot")
                axes[1, 2].set_title(
                    f'发网检测结果\n佩戴发网: {"是" if wearing_hairnet else "否"}\n蓝色区域比例: {mask_ratio:.3f}'
                )
            else:
                axes[1, 2].text(
                    0.5,
                    0.5,
                    "无法进行发网检测",
                    ha="center",
                    va="center",
                    transform=axes[1, 2].transAxes,
                )
                axes[1, 2].set_title("发网检测 (无数据)")
            axes[1, 2].axis("off")

            plt.tight_layout()

            if save_path:
                person_save_path = save_path.replace(
                    ".png", f"_person_{person_idx + 1}.png"
                )
                plt.savefig(person_save_path, dpi=150, bbox_inches="tight")
                print(f"对比结果已保存: {person_save_path}")
            else:
                plt.show()

            plt.close()


def main():
    parser = argparse.ArgumentParser(description="改进的头部ROI提取对比工具")
    parser.add_argument("--image", "-i", type=str, required=True, help="图像路径")
    parser.add_argument("--save", "-s", type=str, help="保存结果的路径")

    args = parser.parse_args()

    extractor = ImprovedHeadROIExtractor()
    extractor.visualize_comparison(args.image, args.save)


if __name__ == "__main__":
    main()

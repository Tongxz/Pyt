#!/usr/bin/env python3
"""
洗手行为检测演示脚本

展示如何使用洗手行为检测功能进行实时检测
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.behavior import BehaviorRecognizer
from src.core.detector import HumanDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HandwashDetectionDemo:
    """洗手检测演示类"""

    def __init__(self):
        """初始化演示系统"""
        logger.info("初始化洗手检测演示系统...")

        # 初始化检测器
        self.human_detector = HumanDetector()
        self.behavior_recognizer = BehaviorRecognizer(
            use_advanced_detection=True, use_mediapipe=True
        )

        # 演示参数
        self.confidence_threshold = 0.6
        self.track_id_counter = 0

        logger.info("洗手检测演示系统初始化完成")

    def detect_handwashing_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        在单帧中检测洗手行为

        Args:
            frame: 输入图像帧

        Returns:
            检测结果列表
        """
        results = []

        # 1. 检测人体
        persons = self.human_detector.detect(frame)

        for i, person in enumerate(persons):
            person_bbox = person["bbox"]

            # 2. 估算手部区域（简化版本）
            hand_regions = self._estimate_hand_regions(person_bbox)

            # 3. 检测洗手行为
            track_id = self.track_id_counter + i
            handwash_confidence = self.behavior_recognizer.detect_handwashing(
                person_bbox, hand_regions, track_id=track_id, frame=frame
            )

            # 4. 检测消毒行为
            sanitize_confidence = self.behavior_recognizer.detect_sanitizing(
                person_bbox, hand_regions, track_id=track_id, frame=frame
            )

            # 5. 整理结果
            result = {
                "person_id": i + 1,
                "person_bbox": person_bbox,
                "hand_regions": hand_regions,
                "handwashing": {
                    "confidence": handwash_confidence,
                    "detected": handwash_confidence > self.confidence_threshold,
                },
                "sanitizing": {
                    "confidence": sanitize_confidence,
                    "detected": sanitize_confidence > self.confidence_threshold,
                },
            }

            results.append(result)

        self.track_id_counter += len(persons)
        return results

    def _estimate_hand_regions(self, person_bbox: List[int]) -> List[Dict]:
        """
        估算手部区域（简化版本）

        Args:
            person_bbox: 人体边界框

        Returns:
            手部区域列表
        """
        x1, y1, x2, y2 = person_bbox
        person_width = x2 - x1
        person_height = y2 - y1

        # 估算左右手位置（身体中部偏下）
        hand_size = min(person_width // 8, person_height // 10)
        hand_y = y1 + int(person_height * 0.6)  # 身体60%高度处

        left_hand_x = x1 + int(person_width * 0.25)
        right_hand_x = x1 + int(person_width * 0.75)

        hand_regions = [
            {
                "bbox": [
                    left_hand_x - hand_size,
                    hand_y - hand_size,
                    left_hand_x + hand_size,
                    hand_y + hand_size,
                ],
                "confidence": 0.8,
                "source": "estimated",
            },
            {
                "bbox": [
                    right_hand_x - hand_size,
                    hand_y - hand_size,
                    right_hand_x + hand_size,
                    hand_y + hand_size,
                ],
                "confidence": 0.8,
                "source": "estimated",
            },
        ]

        return hand_regions

    def draw_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            frame: 输入图像
            results: 检测结果

        Returns:
            标注后的图像
        """
        annotated_frame = frame.copy()

        for result in results:
            person_bbox = result["person_bbox"]
            hand_regions = result["hand_regions"]
            handwashing = result["handwashing"]
            sanitizing = result["sanitizing"]

            # 绘制人体边界框
            x1, y1, x2, y2 = map(int, person_bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制手部区域
            for hand_region in hand_regions:
                hx1, hy1, hx2, hy2 = map(int, hand_region["bbox"])
                cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1)

            # 显示检测结果
            person_id = result["person_id"]
            text_y = y1 - 10

            # 洗手检测结果
            if handwashing["detected"]:
                handwash_text = (
                    f"Person {person_id}: 洗手 ({handwashing['confidence']:.2f})"
                )
                cv2.putText(
                    annotated_frame,
                    handwash_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                text_y -= 25

            # 消毒检测结果
            if sanitizing["detected"]:
                sanitize_text = (
                    f"Person {person_id}: 消毒 ({sanitizing['confidence']:.2f})"
                )
                cv2.putText(
                    annotated_frame,
                    sanitize_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )

        return annotated_frame

    def run_image_demo(self, image_path: str):
        """
        运行图像演示

        Args:
            image_path: 图像文件路径
        """
        logger.info(f"运行图像演示: {image_path}")

        # 读取图像
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"无法读取图像: {image_path}")
            return

        # 检测洗手行为
        start_time = time.time()
        results = self.detect_handwashing_in_frame(frame)
        detection_time = time.time() - start_time

        # 绘制结果
        annotated_frame = self.draw_results(frame, results)

        # 显示结果
        logger.info(f"检测完成，耗时: {detection_time:.3f}s")
        logger.info(f"检测到 {len(results)} 个人")

        for i, result in enumerate(results):
            handwashing = result["handwashing"]
            sanitizing = result["sanitizing"]
            logger.info(
                f"Person {i+1}: 洗手={handwashing['detected']} ({handwashing['confidence']:.3f}), "
                f"消毒={sanitizing['detected']} ({sanitizing['confidence']:.3f})"
            )

        # 保存结果图像
        output_path = image_path.replace(".", "_result.")
        cv2.imwrite(output_path, annotated_frame)
        logger.info(f"结果已保存到: {output_path}")

        # 显示图像（如果在支持的环境中）
        try:
            cv2.imshow("Handwash Detection Demo", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            logger.info("无法显示图像窗口，结果已保存到文件")

    def run_webcam_demo(self):
        """
        运行摄像头实时演示
        """
        logger.info("启动摄像头实时演示...")

        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("无法打开摄像头")
            return

        logger.info("摄像头已打开，按 'q' 退出演示")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("无法读取摄像头帧")
                    break

                # 检测洗手行为
                results = self.detect_handwashing_in_frame(frame)

                # 绘制结果
                annotated_frame = self.draw_results(frame, results)

                # 显示帧率和检测信息
                info_text = f"Persons: {len(results)}"
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # 显示图像
                cv2.imshow("Handwash Detection Demo - Press q to quit", annotated_frame)

                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("摄像头演示结束")

    def cleanup(self):
        """
        清理资源
        """
        if (
            hasattr(self.behavior_recognizer, "pose_detector")
            and self.behavior_recognizer.pose_detector
        ):
            self.behavior_recognizer.pose_detector.cleanup()
        logger.info("演示系统资源已清理")


def main():
    """
    主函数
    """
    import argparse

    parser = argparse.ArgumentParser(description="洗手行为检测演示")
    parser.add_argument(
        "--mode",
        choices=["image", "webcam"],
        default="image",
        help="演示模式: image (图像) 或 webcam (摄像头)",
    )
    parser.add_argument("--image", type=str, help="图像文件路径 (仅在 image 模式下使用)")

    args = parser.parse_args()

    # 创建演示系统
    demo = HandwashDetectionDemo()

    try:
        if args.mode == "image":
            if args.image:
                demo.run_image_demo(args.image)
            else:
                # 使用默认测试图像
                test_image_path = "tests/fixtures/images/test_person.jpg"
                if Path(test_image_path).exists():
                    demo.run_image_demo(test_image_path)
                else:
                    logger.error("请指定图像文件路径: --image <path>")
        elif args.mode == "webcam":
            demo.run_webcam_demo()

    except KeyboardInterrupt:
        logger.info("用户中断演示")
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()

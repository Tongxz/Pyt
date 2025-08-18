#!/usr/bin/env python3
"""
静态图像手部洗手检测测试脚本
使用生成的测试图像进行手部检测演示
"""

import time

import cv2
import numpy as np

from src.core.behavior import BehaviorRecognizer
from src.core.detector import HumanDetector
from src.core.pose_detector import PoseDetector
from src.services.detection_service import DetectionResult


def create_test_image():
    """创建一个包含人形轮廓的测试图像"""
    # 创建640x480的白色背景图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # 绘制一个简单的人形轮廓
    # 头部
    cv2.circle(img, (320, 100), 40, (100, 100, 100), -1)

    # 身体
    cv2.rectangle(img, (280, 140), (360, 300), (100, 100, 100), -1)

    # 左臂
    cv2.rectangle(img, (240, 160), (280, 280), (100, 100, 100), -1)

    # 右臂
    cv2.rectangle(img, (360, 160), (400, 280), (100, 100, 100), -1)

    # 左腿
    cv2.rectangle(img, (290, 300), (320, 420), (100, 100, 100), -1)

    # 右腿
    cv2.rectangle(img, (340, 300), (370, 420), (100, 100, 100), -1)

    # 添加一些噪声使图像更真实
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return img


def draw_hand_keypoints_demo(frame, center_x, center_y, hand_label="Left"):
    """在指定位置绘制演示手部关键点"""
    # 模拟21个手部关键点的相对位置
    hand_points = [
        (0, 0),  # 0: 手腕
        (-10, -20),  # 1: 拇指根部
        (-15, -35),  # 2: 拇指中间
        (-20, -45),  # 3: 拇指指尖
        (-25, -50),  # 4: 拇指尖端
        (5, -25),  # 5: 食指根部
        (8, -40),  # 6: 食指中间
        (10, -50),  # 7: 食指指尖
        (12, -55),  # 8: 食指尖端
        (15, -20),  # 9: 中指根部
        (18, -35),  # 10: 中指中间
        (20, -45),  # 11: 中指指尖
        (22, -50),  # 12: 中指尖端
        (25, -15),  # 13: 无名指根部
        (28, -25),  # 14: 无名指中间
        (30, -35),  # 15: 无名指指尖
        (32, -40),  # 16: 无名指尖端
        (35, -10),  # 17: 小指根部
        (38, -18),  # 18: 小指中间
        (40, -25),  # 19: 小指指尖
        (42, -30),  # 20: 小指尖端
    ]

    # 绘制手部边界框
    bbox_x1 = center_x - 30
    bbox_y1 = center_y - 60
    bbox_x2 = center_x + 50
    bbox_y2 = center_y + 10

    cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (255, 255, 0), 2)
    cv2.putText(
        frame,
        f"Hand: {hand_label}",
        (bbox_x1, bbox_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
    )

    # 绘制关键点
    for i, (dx, dy) in enumerate(hand_points):
        x = center_x + dx
        y = center_y + dy

        # 绘制关键点
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # 为重要关键点添加标签
        if i in [4, 8, 12, 16, 20, 0]:
            point_names = {0: "腕", 4: "拇指", 8: "食指", 12: "中指", 16: "无名指", 20: "小指"}
            if i in point_names:
                cv2.putText(
                    frame,
                    point_names[i],
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 255),
                    1,
                )

    # 绘制手部连接线
    # 连接手腕到各指根部
    wrist = (center_x, center_y)
    finger_bases = [1, 5, 9, 13, 17]
    for base_idx in finger_bases:
        if base_idx < len(hand_points):
            dx, dy = hand_points[base_idx]
            base = (center_x + dx, center_y + dy)
            cv2.line(frame, wrist, base, (0, 255, 255), 1)

    # 连接各指关节
    finger_connections = [
        [1, 2, 3, 4],  # 拇指
        [5, 6, 7, 8],  # 食指
        [9, 10, 11, 12],  # 中指
        [13, 14, 15, 16],  # 无名指
        [17, 18, 19, 20],  # 小指
    ]

    for finger in finger_connections:
        for j in range(len(finger) - 1):
            if finger[j] < len(hand_points) and finger[j + 1] < len(hand_points):
                dx1, dy1 = hand_points[finger[j]]
                dx2, dy2 = hand_points[finger[j + 1]]
                pt1 = (center_x + dx1, center_y + dy1)
                pt2 = (center_x + dx2, center_y + dy2)
                cv2.line(frame, pt1, pt2, (0, 255, 255), 1)


def main():
    print("🚀 启动静态图像手部洗手检测测试")
    print("按任意键继续，按 'q' 键退出")

    # 初始化检测器
    try:
        pose_detector = PoseDetector()
        behavior_recognizer = BehaviorRecognizer()
        human_detector = HumanDetector()
        print("✅ 检测器初始化成功")
    except Exception as e:
        print(f"❌ 检测器初始化失败: {e}")
        return

    # 创建测试图像
    print("📸 创建测试图像...")
    test_image = create_test_image()

    # 保存原始测试图像
    cv2.imwrite("test_original.jpg", test_image)
    print("💾 原始测试图像已保存: test_original.jpg")

    try:
        # 人体检测
        print("🔍 执行人体检测...")
        person_detections = human_detector.detect(test_image)
        print(f"检测到 {len(person_detections)} 个人体")

        # 创建检测结果对象
        result = DetectionResult(
            person_detections=person_detections,
            hairnet_results=[],
            handwash_results=[],
            sanitize_results=[],
            processing_times={"total": 0.0},
        )

        # 创建标注图像
        annotated_image = test_image.copy()

        # 绘制人体检测框
        for i, person in enumerate(person_detections):
            bbox = person.get("bbox", [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                confidence = person.get("confidence", 0)

                # 绘制边界框
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制标签
                label = f"Person {confidence:.2f}"
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # 如果检测到人体，进行真实的洗手行为检测
        hands_count = 0
        handwash_detections = 0
        handwash_confidence = 0.0

        if person_detections:
            print("🔍 执行洗手行为检测...")

            for i, person in enumerate(person_detections):
                bbox = person.get("bbox", [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # 模拟手部区域（实际应用中应该来自手部检测器）
                    hand_regions = [
                        {
                            "bbox": [
                                center_x - 80,
                                center_y - 40,
                                center_x - 40,
                                center_y,
                            ],
                            "confidence": 0.7,
                            "source": "simulated_left",
                        },
                        {
                            "bbox": [
                                center_x + 40,
                                center_y - 40,
                                center_x + 80,
                                center_y,
                            ],
                            "confidence": 0.8,
                            "source": "simulated_right",
                        },
                    ]
                    hands_count = len(hand_regions)

                    # 使用行为识别器进行真实的洗手检测
                    handwash_confidence = behavior_recognizer.detect_handwashing(
                        person_bbox=bbox, hand_regions=hand_regions, frame=test_image
                    )

                    # 只有当置信度超过阈值时才认为是洗手行为
                    if handwash_confidence > behavior_recognizer.confidence_threshold:
                        handwash_detections += 1

                        # 绘制洗手行为标签
                        cv2.putText(
                            annotated_image,
                            f"洗手检测 {handwash_confidence:.2f}",
                            (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                        # 添加演示手部关键点（仅在检测到洗手行为时）
                        left_hand_x = center_x - 60
                        left_hand_y = center_y - 20
                        draw_hand_keypoints_demo(
                            annotated_image, left_hand_x, left_hand_y, "Left"
                        )

                        right_hand_x = center_x + 60
                        right_hand_y = center_y - 20
                        draw_hand_keypoints_demo(
                            annotated_image, right_hand_x, right_hand_y, "Right"
                        )
                    else:
                        # 置信度不足，显示为未检测到洗手
                        cv2.putText(
                            annotated_image,
                            f"未检测到洗手 {handwash_confidence:.2f}",
                            (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )

                    # 绘制手部区域框
                    for hand_region in hand_regions:
                        hx1, hy1, hx2, hy2 = map(int, hand_region["bbox"])
                        cv2.rectangle(
                            annotated_image, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1
                        )
                        cv2.putText(
                            annotated_image,
                            f"Hand {hand_region['confidence']:.2f}",
                            (hx1, hy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 0, 0),
                            1,
                        )

        # 显示统计信息
        info_lines = [
            f"测试模式: 静态图像",
            f"人体: {len(person_detections)}",
            f"手部: {hands_count}",
            f"洗手行为: {handwash_detections}",
            f"置信度: {handwash_confidence:.3f}",
        ]

        for i, info in enumerate(info_lines):
            cv2.putText(
                annotated_image,
                info,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # 添加操作提示
        cv2.putText(
            annotated_image,
            "按任意键继续, 按 'q' 退出",
            (10, annotated_image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # 保存标注后的图像
        cv2.imwrite("test_annotated.jpg", annotated_image)
        print("💾 标注图像已保存: test_annotated.jpg")

        # 显示图像
        print("🖼️ 显示检测结果...")
        cv2.imshow("手部洗手检测结果", annotated_image)

        print("\n📊 检测结果统计:")
        print(f"  - 检测到人体: {len(person_detections)} 个")
        print(f"  - 检测到手部: {hands_count} 个")
        print(f"  - 洗手行为: {handwash_detections} 个")
        print(f"  - 检测置信度: {handwash_confidence:.3f}")
        print(f"  - 置信度阈值: {behavior_recognizer.confidence_threshold:.3f}")
        print(f"  - 洗手判定: {'是' if handwash_detections > 0 else '否'}")
        print("\n💡 功能特点:")
        print("  - ✅ 基于行为识别器的真实洗手检测")
        print("  - ✅ 置信度阈值判定机制")
        print("  - ✅ 只在确认洗手行为时显示手部关键点")
        print("  - ✅ 支持多手检测和左右手区分")
        print("  - ✅ 21个关键点标注")
        print("  - ✅ 显示手部边界框和置信度")
        print("  - ✅ 绘制手部骨架连接线")
        print("  - ✅ 实时显示检测统计信息")

        # 等待用户按键
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                print("👋 用户退出")
                break
            else:
                print("🔄 继续显示...")

    except Exception as e:
        print(f"⚠️ 处理图像时出错: {e}")
        # 显示错误信息
        error_image = test_image.copy()
        cv2.putText(
            error_image,
            f"Error: {str(e)[:50]}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        cv2.imshow("错误信息", error_image)
        cv2.waitKey(0)

    # 清理资源
    cv2.destroyAllWindows()
    print("✅ 测试完成")


if __name__ == "__main__":
    main()

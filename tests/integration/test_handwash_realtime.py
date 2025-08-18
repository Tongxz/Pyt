#!/usr/bin/env python3
"""
实时手部洗手检测测试脚本
使用摄像头进行实时检测，显示手部关键点和洗手行为识别
"""

import time

import cv2
import numpy as np

from src.core.behavior import BehaviorRecognizer
from src.core.detector import HumanDetector
from src.core.pose_detector import PoseDetector
from src.services.detection_service import DetectionResult


def main():
    print("🚀 启动实时手部洗手检测测试")
    print("按 'q' 键退出，按 's' 键截图保存")

    # 初始化检测器
    try:
        pose_detector = PoseDetector()
        behavior_recognizer = BehaviorRecognizer()
        human_detector = HumanDetector()
        print("✅ 检测器初始化成功")
    except Exception as e:
        print(f"❌ 检测器初始化失败: {e}")
        return

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("📹 摄像头已启动")

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头帧")
            break

        frame_count += 1
        fps_counter += 1

        # 计算FPS
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        try:
            # 人体检测
            person_detections = human_detector.detect(frame)

            # 创建检测结果对象
            result = DetectionResult(
                person_detections=person_detections,
                hairnet_results=[],
                handwash_results=[],
                sanitize_results=[],
            )

            # 只有在检测到人体时才进行手部检测
            hands_count = 0
            if person_detections:
                # 手部关键点检测
                hands_results = pose_detector.detect_hands(frame)
                hands_count = len(hands_results)

                # 洗手行为检测
                for i, person in enumerate(person_detections):
                    person_bbox = person.get("bbox", [])
                    if len(person_bbox) >= 4:
                        # 使用行为识别器检测洗手动作
                        handwash_result = behavior_recognizer.detect_handwashing(
                            frame, person_bbox, hands_results
                        )

                        if handwash_result:
                            result.handwash_results.append(
                                {
                                    "person_bbox": person_bbox,
                                    "is_handwashing": handwash_result.get(
                                        "is_handwashing", False
                                    ),
                                    "confidence": handwash_result.get(
                                        "confidence", 0.0
                                    ),
                                }
                            )

                # 绘制手部关键点
                for hand_result in hands_results:
                    landmarks = hand_result["landmarks"]
                    hand_label = hand_result["label"]
                    bbox = hand_result["bbox"]

                    # 绘制手部边界框
                    cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2
                    )  # 黄色边界框

                    # 绘制手部标签
                    cv2.putText(
                        frame,
                        f"Hand: {hand_label}",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )

                    # 绘制手部关键点
                    h, w = frame.shape[:2]
                    for i, landmark in enumerate(landmarks):
                        x = int(landmark["x"] * w)
                        y = int(landmark["y"] * h)

                        # 绘制关键点
                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # 青色圆点

                        # 为重要关键点添加标签
                        if i in [4, 8, 12, 16, 20, 0]:  # MediaPipe手部关键点索引
                            point_names = {
                                0: "腕",
                                4: "拇指",
                                8: "食指",
                                12: "中指",
                                16: "无名指",
                                20: "小指",
                            }
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
                    if len(landmarks) >= 21:
                        # 连接手腕到各指根部
                        wrist = (int(landmarks[0]["x"] * w), int(landmarks[0]["y"] * h))
                        finger_bases = [5, 9, 13, 17]
                        for base_idx in finger_bases:
                            if base_idx < len(landmarks):
                                base = (
                                    int(landmarks[base_idx]["x"] * w),
                                    int(landmarks[base_idx]["y"] * h),
                                )
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
                                if finger[j] < len(landmarks) and finger[j + 1] < len(
                                    landmarks
                                ):
                                    pt1 = (
                                        int(landmarks[finger[j]]["x"] * w),
                                        int(landmarks[finger[j]]["y"] * h),
                                    )
                                    pt2 = (
                                        int(landmarks[finger[j + 1]]["x"] * w),
                                        int(landmarks[finger[j + 1]]["y"] * h),
                                    )
                                    cv2.line(frame, pt1, pt2, (0, 255, 255), 1)

            # 绘制人体检测框
            for person in person_detections:
                bbox = person.get("bbox", [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    confidence = person.get("confidence", 0)

                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 绘制标签
                    label = f"Person {confidence:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # 绘制洗手检测结果
            for handwash_result in result.handwash_results:
                if handwash_result.get("is_handwashing", False):
                    person_bbox = handwash_result.get("person_bbox", [])
                    if len(person_bbox) >= 4:
                        x1, y1, x2, y2 = map(int, person_bbox[:4])
                        confidence = handwash_result.get("confidence", 0)

                        # 绘制洗手标签
                        cv2.putText(
                            frame,
                            f"洗手中 {confidence:.2f}",
                            (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

            # 显示统计信息
            info_lines = [
                f"FPS: {current_fps}",
                f"帧数: {frame_count}",
                f"人体: {len(person_detections)}",
                f"手部: {hands_count}",
                f"洗手: {len([r for r in result.handwash_results if r.get('is_handwashing', False)])}",
            ]

            for i, info in enumerate(info_lines):
                cv2.putText(
                    frame,
                    info,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            # 添加操作提示
            cv2.putText(
                frame,
                "按 'q' 退出, 按 's' 截图",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        except Exception as e:
            print(f"⚠️ 处理帧时出错: {e}")
            # 在帧上显示错误信息
            cv2.putText(
                frame,
                f"Error: {str(e)[:50]}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        # 显示帧
        cv2.imshow("实时手部洗手检测", frame)

        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("👋 用户退出")
            break
        elif key == ord("s"):
            # 保存截图
            timestamp = int(time.time())
            filename = f"handwash_detection_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 截图已保存: {filename}")

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ 测试完成，共处理 {frame_count} 帧")


if __name__ == "__main__":
    main()

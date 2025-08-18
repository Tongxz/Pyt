#!/usr/bin/env python3
"""
使用真实测试文件进行手部洗手检测测试
使用 tests/fixtures 目录下的实际图片和视频文件
"""

import os
import time

import cv2
import numpy as np

from src.core.behavior import BehaviorRecognizer
from src.core.detector import HumanDetector
from src.core.pose_detector import PoseDetector
from src.services.detection_service import DetectionResult


def process_image(image_path, detectors):
    """处理单张图片"""
    pose_detector, behavior_recognizer, human_detector = detectors

    print(f"\n📸 处理图片: {os.path.basename(image_path)}")

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None

    print(f"   图片尺寸: {image.shape[1]}x{image.shape[0]}")

    try:
        # 人体检测
        person_detections = human_detector.detect(image)
        print(f"   检测到人体: {len(person_detections)} 个")

        # 创建检测结果对象
        result = DetectionResult(
            person_detections=person_detections,
            hairnet_results=[],
            handwash_results=[],
            sanitize_results=[],
            processing_times={"total": 0.0},
        )

        # 创建标注图像
        annotated_image = image.copy()
        hands_count = 0

        # 只有在检测到人体时才进行手部检测
        if person_detections:
            print("   🔍 执行手部关键点检测...")

            # 手部关键点检测
            hands_results = pose_detector.detect_hands(image)
            hands_count = len(hands_results)
            print(f"   检测到手部: {hands_count} 个")

            # 洗手行为检测
            for i, person in enumerate(person_detections):
                person_bbox = person.get("bbox", [])
                if len(person_bbox) >= 4:
                    # 使用行为识别器检测洗手动作
                    handwash_confidence = behavior_recognizer.detect_handwashing(
                        person_bbox, hands_results, track_id=i, frame=image
                    )

                    handwash_result = {
                        "is_handwashing": handwash_confidence
                        > behavior_recognizer.confidence_threshold,
                        "confidence": handwash_confidence,
                    }

                    if handwash_result:
                        result.handwash_results.append(
                            {
                                "person_bbox": person_bbox,
                                "is_handwashing": handwash_result.get(
                                    "is_handwashing", False
                                ),
                                "confidence": handwash_result.get("confidence", 0.0),
                            }
                        )

            # 绘制手部关键点
            for hand_result in hands_results:
                landmarks = hand_result["landmarks"]
                hand_label = hand_result["label"]
                bbox = hand_result["bbox"]

                # 绘制手部边界框
                cv2.rectangle(
                    annotated_image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (255, 255, 0),
                    2,
                )  # 黄色边界框

                # 绘制手部标签
                cv2.putText(
                    annotated_image,
                    f"Hand: {hand_label}",
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )

                # 绘制手部关键点
                h, w = annotated_image.shape[:2]
                for i, landmark in enumerate(landmarks):
                    x = int(landmark["x"] * w)
                    y = int(landmark["y"] * h)

                    # 绘制关键点
                    cv2.circle(annotated_image, (x, y), 3, (0, 255, 255), -1)  # 青色圆点

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
                                annotated_image,
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
                            cv2.line(annotated_image, wrist, base, (0, 255, 255), 1)

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
                                cv2.line(annotated_image, pt1, pt2, (0, 255, 255), 1)

        # 绘制人体检测框
        for person in person_detections:
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

        # 绘制洗手检测结果
        for handwash_result in result.handwash_results:
            if handwash_result.get("is_handwashing", False):
                person_bbox = handwash_result.get("person_bbox", [])
                if len(person_bbox) >= 4:
                    x1, y1, x2, y2 = map(int, person_bbox[:4])
                    confidence = handwash_result.get("confidence", 0)

                    # 绘制洗手标签
                    cv2.putText(
                        annotated_image,
                        f"洗手中 {confidence:.2f}",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

        # 显示统计信息
        info_lines = [
            f"文件: {os.path.basename(image_path)}",
            f"人体: {len(person_detections)}",
            f"手部: {hands_count}",
            f"洗手: {len([r for r in result.handwash_results if r.get('is_handwashing', False)])}",
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
            "按任意键继续, 按 'q' 退出, 按 's' 保存",
            (10, annotated_image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return annotated_image, result

    except Exception as e:
        print(f"   ⚠️ 处理图片时出错: {e}")
        return None, None


def process_video(video_path, detectors, max_frames=100):
    """处理视频文件"""
    pose_detector, behavior_recognizer, human_detector = detectors

    print(f"\n🎥 处理视频: {os.path.basename(video_path)}")

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"   视频信息: {width}x{height}, {fps:.1f}fps, {frame_count}帧")
    print(f"   将处理前 {min(max_frames, frame_count)} 帧")

    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret or processed_frames >= max_frames:
            break

        frame_idx += 1

        # 每隔几帧处理一次以提高性能
        if frame_idx % 5 != 0:
            continue

        processed_frames += 1

        try:
            # 人体检测
            person_detections = human_detector.detect(frame)
            hands_count = 0

            # 只有在检测到人体时才进行手部检测
            if person_detections:
                # 手部关键点检测
                hands_results = pose_detector.detect_hands(frame)
                hands_count = len(hands_results)

                # 绘制手部关键点
                for hand_result in hands_results:
                    landmarks = hand_result["landmarks"]
                    hand_label = hand_result["label"]
                    bbox = hand_result["bbox"]

                    # 绘制手部边界框
                    cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2
                    )

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
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                        # 为重要关键点添加标签
                        if i in [0, 4, 8, 12, 16, 20]:
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
                                    (x + 3, y - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.3,
                                    (0, 255, 255),
                                    1,
                                )

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

            # 显示统计信息
            info_lines = [
                f"帧: {frame_idx}/{frame_count}",
                f"人体: {len(person_detections)}",
                f"手部: {hands_count}",
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
            print(f"   ⚠️ 处理第{frame_idx}帧时出错: {e}")

        # 显示帧
        cv2.imshow(f"视频检测 - {os.path.basename(video_path)}", frame)

        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("   👋 用户退出视频播放")
            break
        elif key == ord("s"):
            # 保存截图
            timestamp = int(time.time())
            filename = f"video_frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"   📸 截图已保存: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"   ✅ 视频处理完成，共处理 {processed_frames} 帧")


def main():
    print("🚀 启动真实测试文件手部洗手检测")
    print("使用 tests/fixtures 目录下的实际图片和视频文件")

    # 初始化检测器
    try:
        pose_detector = PoseDetector()
        behavior_recognizer = BehaviorRecognizer()
        human_detector = HumanDetector()
        detectors = (pose_detector, behavior_recognizer, human_detector)
        print("✅ 检测器初始化成功")
    except Exception as e:
        print(f"❌ 检测器初始化失败: {e}")
        return

    # 测试文件路径
    fixtures_dir = "/Users/zhou/Code/python/Pyt/tests/fixtures"
    image_files = [
        os.path.join(fixtures_dir, "images/person/test_person.png"),
        os.path.join(fixtures_dir, "images/hairnet/7月23日.png"),
    ]
    video_files = [
        os.path.join(fixtures_dir, "videos/20250724072708.mp4"),
        os.path.join(fixtures_dir, "videos/20250724072822_175680.mp4"),
    ]

    print("\n📋 测试计划:")
    print(f"  - 图片文件: {len([f for f in image_files if os.path.exists(f)])} 个")
    print(f"  - 视频文件: {len([f for f in video_files if os.path.exists(f)])} 个")

    # 处理图片文件
    print("\n🖼️ ===== 图片检测测试 =====")
    for image_path in image_files:
        if os.path.exists(image_path):
            annotated_image, result = process_image(image_path, detectors)

            if annotated_image is not None:
                # 保存标注后的图像
                output_filename = f"annotated_{os.path.basename(image_path)}"
                cv2.imwrite(output_filename, annotated_image)
                print(f"   💾 标注图像已保存: {output_filename}")

                # 显示图像
                cv2.imshow(f"检测结果 - {os.path.basename(image_path)}", annotated_image)

                print(f"\n   📊 检测结果统计:")
                if result:
                    print(f"     - 检测到人体: {len(result.person_detections)} 个")
                    hands_count = 0
                    try:
                        hands_results = pose_detector.detect_hands(
                            cv2.imread(image_path)
                        )
                        hands_count = len(hands_results)
                    except:
                        pass
                    print(f"     - 检测到手部: {hands_count} 个")
                    print(
                        f"     - 洗手行为: {len([r for r in result.handwash_results if r.get('is_handwashing', False)])} 个"
                    )

                print("\n   按任意键继续下一张图片，按 'q' 跳过图片测试...")
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()

                if key == ord("q"):
                    print("   👋 跳过剩余图片测试")
                    break
        else:
            print(f"   ⚠️ 图片文件不存在: {image_path}")

    # 询问是否继续视频测试
    print("\n🎥 ===== 视频检测测试 =====")
    print("是否继续进行视频检测测试？(y/n): ", end="")
    choice = input().lower().strip()

    if choice == "y" or choice == "yes":
        # 处理视频文件
        for video_path in video_files:
            if os.path.exists(video_path):
                process_video(video_path, detectors, max_frames=50)  # 限制处理帧数

                print("\n继续下一个视频？(y/n): ", end="")
                choice = input().lower().strip()
                if choice != "y" and choice != "yes":
                    print("👋 跳过剩余视频测试")
                    break
            else:
                print(f"⚠️ 视频文件不存在: {video_path}")
    else:
        print("👋 跳过视频测试")

    print("\n✅ 测试完成")
    print("\n💡 功能特点:")
    print("  - ✅ 使用真实测试文件进行检测")
    print("  - ✅ 只在检测到人体时显示手部关键点")
    print("  - ✅ 支持多手检测和左右手区分")
    print("  - ✅ 21个关键点标注")
    print("  - ✅ 显示手部边界框和重要关键点标签")
    print("  - ✅ 绘制手部骨架连接线")
    print("  - ✅ 实时显示检测统计信息")
    print("  - ✅ 支持图片和视频文件处理")
    print("  - ✅ 自动保存标注结果")


if __name__ == "__main__":
    main()

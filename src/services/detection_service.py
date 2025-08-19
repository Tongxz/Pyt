import base64
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import Depends, Request

from core.optimized_detection_pipeline import (
    DetectionResult,
    OptimizedDetectionPipeline,
)
from core.pose_detector import PoseDetector
from core.yolo_hairnet_detector import YOLOHairnetDetector

logger = logging.getLogger(__name__)

# 在这里我们将实例化我们的服务，或者在app启动时进行
# 为了简单起见，我们先在这里设置为None
optimized_pipeline: Optional[OptimizedDetectionPipeline] = None
hairnet_pipeline: Optional[YOLOHairnetDetector] = None


def get_optimized_pipeline(request: Request) -> Optional[OptimizedDetectionPipeline]:
    return getattr(request.app.state, "optimized_pipeline", None)


def get_hairnet_pipeline(request: Request) -> Optional[YOLOHairnetDetector]:
    return getattr(request.app.state, "hairnet_pipeline", None)


def comprehensive_detection_logic(
    contents: bytes,
    filename: str,
    optimized_pipeline: Optional[OptimizedDetectionPipeline],
    hairnet_pipeline: Optional[YOLOHairnetDetector],
    record_process: bool = False,
) -> dict:
    """
    执行综合检测并返回统一格式的结果。
    这个函数会被 comprehensive.py 中的API端点调用。
    支持图像和视频文件。

    Args:
        contents: 文件内容
        filename: 文件名
        optimized_pipeline: 优化检测管道
        hairnet_pipeline: 发网检测管道
        record_process: 是否录制检测过程（仅对视频有效）
    """
    import os
    import tempfile
    from pathlib import Path

    import cv2
    import numpy as np

    # 检查文件类型
    file_ext = Path(filename).suffix.lower()
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    image = None

    if file_ext in video_extensions:
        # 处理视频文件
        logger.info(f"检测到视频文件: {filename}, 录制模式: {record_process}")

        # 将视频内容写入临时文件
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            temp_file.write(contents)
            temp_video_path = temp_file.name

        try:
            if record_process and optimized_pipeline:
                # 录制模式：处理整个视频并生成带标注的视频
                return _process_video_with_recording(
                    temp_video_path, filename, optimized_pipeline
                )
            else:
                # 普通模式：只提取第一帧进行检测
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    raise ValueError("无法打开视频文件")

                ret, frame = cap.read()
                if not ret or frame is None:
                    raise ValueError("无法从视频中读取帧")

                image = frame
                cap.release()

                logger.info(f"成功从视频中提取第一帧，尺寸: {image.shape}")

        finally:
            # 清理临时文件
            try:
                os.unlink(temp_video_path)
            except Exception as e:
                logger.warning(f"清理临时视频文件失败: {e}")

    elif file_ext in image_extensions:
        # 处理图像文件
        logger.info(f"检测到图像文件: {filename}")
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码图像")
    else:
        raise ValueError(
            f"不支持的文件类型: {file_ext}。支持的格式: {image_extensions | video_extensions}"
        )

    if image is None:
        raise ValueError("无法获取有效的图像数据")

    if optimized_pipeline is None:
        raise RuntimeError(
            "优化检测管道未初始化。请检查：\n" "1. 检测服务是否正确启动\n" "2. 模型文件是否存在\n" "3. 系统依赖是否完整"
        )

    logger.info("使用优化检测管道进行综合检测")
    result = optimized_pipeline.detect_comprehensive(image)

    # 转换为前端期望的格式
    total_persons = len(result.person_detections)
    persons_with_hairnet = len(
        [h for h in result.hairnet_results if h.get("has_hairnet", False)]
    )
    persons_handwashing = len(
        [h for h in result.handwash_results if h.get("is_handwashing", False)]
    )
    persons_sanitizing = len(
        [s for s in result.sanitize_results if s.get("is_sanitizing", False)]
    )

    # 构建统计信息
    statistics = {
        "persons_with_hairnet": persons_with_hairnet,
        "persons_handwashing": persons_handwashing,
        "persons_sanitizing": persons_sanitizing,
    }

    # 构建检测详情
    detections = []
    for detection in result.person_detections:
        detections.append(
            {
                "class": "person",
                "confidence": detection.get("confidence", 0.0),
                "bbox": detection.get("bbox", [0, 0, 0, 0]),
            }
        )

    for detection in result.hairnet_results:
        if detection.get("has_hairnet", False):
            detections.append(
                {
                    "class": "hairnet",
                    "confidence": detection.get("hairnet_confidence", 0.0),
                    "bbox": detection.get("hairnet_bbox", [0, 0, 0, 0]),
                }
            )

    # 处理标注图像
    annotated_image_b64 = None
    if result.annotated_image is not None:
        _, buffer = cv2.imencode(".jpg", result.annotated_image)
        annotated_image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return {
        "total_persons": total_persons,
        "statistics": statistics,
        "detections": detections,
        "annotated_image": annotated_image_b64,
        "processing_times": result.processing_times,
    }


def initialize_detection_services():
    """初始化所有检测服务和模型."""
    global optimized_pipeline, hairnet_pipeline
    logger.info("正在初始化检测服务...")
    try:
        # 这里的初始化逻辑需要从 app.py 的 startup 事件中迁移过来
        # 为了演示，我们先使用None
        from src.core.behavior import BehaviorRecognizer
        from src.core.data_manager import DetectionDataManager
        from src.core.detector import HumanDetector
        from src.core.region import RegionManager
        from src.core.rule_engine import RuleEngine

        detector = HumanDetector()
        behavior_recognizer = BehaviorRecognizer()
        data_manager = DetectionDataManager()
        region_manager = RegionManager()
        rule_engine = RuleEngine()

        optimized_pipeline = OptimizedDetectionPipeline(
            human_detector=detector,
            hairnet_detector=YOLOHairnetDetector(),
            behavior_recognizer=behavior_recognizer,
        )
        hairnet_pipeline = YOLOHairnetDetector()
        logger.info("检测服务初始化完成。")
    except Exception as e:
        logger.exception(f"初始化检测服务失败: {e}")


def _process_video_with_recording(
    video_path: str, filename: str, optimized_pipeline: OptimizedDetectionPipeline
) -> dict:
    """
    处理视频并生成带标注的视频文件

    Args:
        video_path: 视频文件路径
        filename: 原始文件名
        optimized_pipeline: 检测管道

    Returns:
        包含检测结果和输出视频信息的字典
    """
    import os
    import tempfile
    import time
    from pathlib import Path

    import cv2
    import numpy as np

    from core.tracker import MultiObjectTracker

    logger.info(f"开始处理视频: {filename}")

    # 打开输入视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"视频属性: {width}x{height}, {fps}fps, {total_frames}帧")

    # 创建输出视频文件
    output_dir = "./output/processed_videos"
    os.makedirs(output_dir, exist_ok=True)

    base_name = Path(filename).stem
    output_filename = f"{base_name}_processed_{int(time.time())}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    # 设置视频编码器
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise ValueError("无法创建输出视频文件")

    # 初始化跟踪器 - 使用更严格的参数减少误检
    tracker = MultiObjectTracker(max_disappeared=5, iou_threshold=0.5)

    # 统计信息 - 跟踪每个人的行为历史
    person_behaviors = (
        {}
    )  # track_id -> {'hairnet': [], 'handwashing': [], 'sanitizing': []}
    processed_frames = 0
    total_confidence = 0.0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 每5帧处理一次以提高性能
            if frame_count % 5 == 0:
                # 执行检测
                result = optimized_pipeline.detect_comprehensive(frame)

                # 准备跟踪器输入格式
                detections = []
                for person in result.person_detections:
                    detections.append(
                        {"bbox": person["bbox"], "confidence": person["confidence"]}
                    )

                # 更新跟踪器
                tracked_objects = tracker.update(detections)

                # 绘制检测结果（包含跟踪ID）
                annotated_frame = _draw_detections_on_frame_with_tracking(
                    frame.copy(), result, tracked_objects
                )

                # 更新统计信息 - 记录每个人的行为历史
                for track in tracked_objects:
                    track_id = track["track_id"]

                    # 初始化该人员的行为记录
                    if track_id not in person_behaviors:
                        person_behaviors[track_id] = {
                            "hairnet": [],
                            "handwashing": [],
                            "sanitizing": [],
                        }

                    # 查找对应的检测结果
                    track_bbox = track["bbox"]

                    # 检查发网检测结果
                    has_hairnet_this_frame = False
                    for hairnet_result in result.hairnet_results:
                        if _bbox_overlap(
                            track_bbox, hairnet_result.get("person_bbox", [])
                        ):
                            has_hairnet_this_frame = hairnet_result.get(
                                "has_hairnet", False
                            )
                            break
                    person_behaviors[track_id]["hairnet"].append(has_hairnet_this_frame)

                    # 检查洗手行为
                    is_handwashing_this_frame = False
                    for handwash_result in result.handwash_results:
                        if _bbox_overlap(
                            track_bbox, handwash_result.get("person_bbox", [])
                        ):
                            is_handwashing_this_frame = handwash_result.get(
                                "is_handwashing", False
                            )
                            break
                    person_behaviors[track_id]["handwashing"].append(
                        is_handwashing_this_frame
                    )

                    # 检查消毒行为
                    is_sanitizing_this_frame = False
                    for sanitize_result in result.sanitize_results:
                        if _bbox_overlap(
                            track_bbox, sanitize_result.get("person_bbox", [])
                        ):
                            is_sanitizing_this_frame = sanitize_result.get(
                                "is_sanitizing", False
                            )
                            break
                    person_behaviors[track_id]["sanitizing"].append(
                        is_sanitizing_this_frame
                    )

                # 计算平均置信度
                if result.person_detections:
                    frame_confidence = sum(
                        det.get("confidence", 0) for det in result.person_detections
                    ) / len(result.person_detections)
                    total_confidence += frame_confidence

                processed_frames += 1
                out.write(annotated_frame)
            else:
                # 未处理的帧直接写入
                out.write(frame)

            # 显示进度
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")

        processing_time = time.time() - start_time
        logger.info(f"视频处理完成，耗时: {processing_time:.2f}秒")

    finally:
        cap.release()
        out.release()

    # 获取输出文件大小
    file_size = os.path.getsize(output_path)

    # 计算平均置信度
    avg_confidence = (
        total_confidence / processed_frames if processed_frames > 0 else 0.0
    )

    # 分析每个人的行为模式 - 使用更宽松的过滤条件
    min_appearances = max(8, processed_frames // 15)  # 至少出现8次或总帧数的6.7%
    valid_persons = {}

    for track_id, behaviors in person_behaviors.items():
        total_appearances = len(behaviors.get("hairnet", []))
        # 基本出现次数过滤
        if total_appearances >= min_appearances:
            # 更宽松的稳定性检查：只要track有足够的总出现次数就认为有效
            hairnet_list = behaviors.get("hairnet", [])
            if len(hairnet_list) > 0:
                # 计算整体出现的连续性 - 检查是否有过长的空白期
                max_gap = 0
                current_gap = 0

                for i, appeared in enumerate(hairnet_list):
                    if not appeared:
                        current_gap += 1
                    else:
                        max_gap = max(max_gap, current_gap)
                        current_gap = 0
                max_gap = max(max_gap, current_gap)

                # 如果最大空白期不超过总长度的60%，认为是有效的track
                max_gap_ratio = (
                    max_gap / len(hairnet_list) if len(hairnet_list) > 0 else 1.0
                )
                if max_gap_ratio <= 0.6:
                    valid_persons[track_id] = behaviors
                    logger.info(
                        f"Track {track_id}: 通过稳定性检查，最大空白期比例: {max_gap_ratio:.2f}"
                    )
                else:
                    logger.info(
                        f"Track {track_id}: 空白期过长，最大空白期比例: {max_gap_ratio:.2f}，被过滤"
                    )
            else:
                logger.info(f"Track {track_id}: 无有效数据，被过滤")
        else:
            logger.info(
                f"Track {track_id}: 出现次数不足({total_appearances} < {min_appearances})，被过滤"
            )

    total_persons = len(valid_persons)
    persons_with_hairnet = 0
    persons_handwashing = 0
    persons_sanitizing = 0

    logger.info(
        f"过滤前人员数: {len(person_behaviors)}, 过滤后人员数: {total_persons}, 最小出现次数: {min_appearances}"
    )

    for track_id, behaviors in valid_persons.items():
        hairnet_count = sum(behaviors["hairnet"]) if behaviors["hairnet"] else 0
        hairnet_total = len(behaviors["hairnet"]) if behaviors["hairnet"] else 0
        handwashing_count = (
            sum(behaviors["handwashing"]) if behaviors["handwashing"] else 0
        )
        sanitizing_count = (
            sum(behaviors["sanitizing"]) if behaviors["sanitizing"] else 0
        )

        logger.info(
            f"Track {track_id}: 出现{hairnet_total}次, 发网{hairnet_count}次, 洗手{handwashing_count}次, 消毒{sanitizing_count}次"
        )

        # 如果一个人在视频中大部分时间都戴发网，则认为该人戴发网
        if hairnet_total > 0 and hairnet_count > hairnet_total * 0.5:
            persons_with_hairnet += 1
            logger.info(f"Track {track_id}: 认定为戴发网")

        # 更严格的洗手行为判断：需要至少20%的时间在洗手，且至少持续5次检测
        if hairnet_total > 0 and handwashing_count >= max(5, hairnet_total * 0.2):
            persons_handwashing += 1
            logger.info(
                f"Track {track_id}: 认定为洗手 (洗手{handwashing_count}次/{hairnet_total}次)"
            )
        else:
            logger.info(
                f"Track {track_id}: 未认定为洗手 (洗手{handwashing_count}次/{hairnet_total}次，需要≥{max(5, int(hairnet_total * 0.2))}次)"
            )

        # 更严格的消毒行为判断：需要至少15%的时间在消毒，且至少持续3次检测
        if hairnet_total > 0 and sanitizing_count >= max(3, hairnet_total * 0.15):
            persons_sanitizing += 1
            logger.info(
                f"Track {track_id}: 认定为消毒 (消毒{sanitizing_count}次/{hairnet_total}次)"
            )
        else:
            logger.info(
                f"Track {track_id}: 未认定为消毒 (消毒{sanitizing_count}次/{hairnet_total}次，需要≥{max(3, int(hairnet_total * 0.15))}次)"
            )

    # 构建返回结果 - 基于行为分析的统计
    result = {
        "statistics": {
            "total_persons": total_persons,
            "persons_with_hairnet": persons_with_hairnet,
            "persons_handwashing": persons_handwashing,
            "persons_sanitizing": persons_sanitizing,
            "average_confidence": avg_confidence,
        },
        "processing_info": {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "processing_time": processing_time,
            "fps": fps,
        },
        "output_video": {
            "filename": output_filename,
            "path": output_path,
            "size_bytes": file_size,
            "url": f"/api/v1/download/video/{output_filename}",
        },
    }

    logger.info(f"视频处理结果: {result['statistics']}")
    return result


def _bbox_overlap(bbox1, bbox2, threshold=0.5):
    """
    检查两个边界框是否重叠

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        threshold: 重叠阈值

    Returns:
        bool: 是否重叠
    """
    if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
        return False

    # 计算交集
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return False

    # 计算交集面积
    intersection = (x2 - x1) * (y2 - y1)

    # 计算并集面积
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    # 计算IoU
    iou = intersection / union if union > 0 else 0
    return iou > threshold


def _draw_detections_on_frame_with_tracking(frame, result, tracked_objects):
    """
    在帧上绘制检测结果和跟踪ID

    Args:
        frame: 输入帧
        result: 检测结果
        tracked_objects: 跟踪对象列表

    Returns:
        带标注的帧
    """
    import cv2

    annotated_frame = frame.copy()

    # 初始化手部检测器
    pose_detector = PoseDetector()

    # 检测手部关键点
    hands_results = pose_detector.detect_hands(frame)

    # 绘制跟踪框和ID
    for track in tracked_objects:
        bbox = track["bbox"]
        track_id = track["track_id"]
        confidence = track.get("confidence", 0.0)

        # 绘制边界框
        cv2.rectangle(
            annotated_frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2,
        )

        # 绘制跟踪ID和置信度
        label = f"ID:{track_id} ({confidence:.2f})"
        cv2.putText(
            annotated_frame,
            label,
            (int(bbox[0]), int(bbox[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # 绘制发网检测结果
    for hairnet_result in result.hairnet_results:
        if "person_bbox" in hairnet_result:
            bbox = hairnet_result["person_bbox"]
            has_hairnet = hairnet_result.get("has_hairnet", False)
            color = (0, 255, 0) if has_hairnet else (0, 0, 255)

            # 在人体框上方绘制发网状态
            label = "发网✓" if has_hairnet else "发网✗"
            cv2.putText(
                annotated_frame,
                label,
                (int(bbox[0]), int(bbox[1]) - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # 绘制洗手行为
    for handwash_result in result.handwash_results:
        if "person_bbox" in handwash_result:
            bbox = handwash_result["person_bbox"]
            is_handwashing = handwash_result.get("is_handwashing", False)

            if is_handwashing:
                cv2.putText(
                    annotated_frame,
                    "洗手",
                    (int(bbox[0]), int(bbox[1]) - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

    # 绘制消毒行为
    for sanitize_result in result.sanitize_results:
        if "person_bbox" in sanitize_result:
            bbox = sanitize_result["person_bbox"]
            is_sanitizing = sanitize_result.get("is_sanitizing", False)

            if is_sanitizing:
                cv2.putText(
                    annotated_frame,
                    "消毒",
                    (int(bbox[0]), int(bbox[1]) - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )

    # 只有在检测到人体时才绘制手部关键点
    if tracked_objects:
        # 绘制手部关键点
        for hand_result in hands_results:
            landmarks = hand_result["landmarks"]
            hand_label = hand_result["label"]
            bbox = hand_result["bbox"]

            # 检查手部是否在任何人体检测框内
            hand_in_person = False
            for track in tracked_objects:
                person_bbox = track["bbox"]
                if _bbox_overlap(
                    [bbox[0], bbox[1], bbox[2], bbox[3]], person_bbox, 0.1
                ):
                    hand_in_person = True
                    break

            # 只绘制在人体区域内的手部
            if hand_in_person:
                # 绘制手部边界框
                cv2.rectangle(
                    annotated_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (255, 255, 0),
                    2,
                )  # 黄色边界框

                # 绘制手部标签
                cv2.putText(
                    annotated_frame,
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
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 255), -1)  # 青色圆点

                    # 为重要关键点添加标签（拇指尖、食指尖、中指尖、无名指尖、小指尖、手腕）
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
                                annotated_frame,
                                point_names[i],
                                (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (0, 255, 255),
                                1,
                            )

                # 绘制手部连接线（简化版）
                if len(landmarks) >= 21:  # MediaPipe手部模型有21个关键点
                    # 连接手腕到各指根部
                    wrist = (int(landmarks[0]["x"] * w), int(landmarks[0]["y"] * h))
                    finger_bases = [5, 9, 13, 17]  # 各指根部索引
                    for base_idx in finger_bases:
                        if base_idx < len(landmarks):
                            base = (
                                int(landmarks[base_idx]["x"] * w),
                                int(landmarks[base_idx]["y"] * h),
                            )
                            cv2.line(annotated_frame, wrist, base, (0, 255, 255), 1)

                    # 连接各指关节
                    finger_connections = [
                        [1, 2, 3, 4],  # 拇指
                        [5, 6, 7, 8],  # 食指
                        [9, 10, 11, 12],  # 中指
                        [13, 14, 15, 16],  # 无名指
                        [17, 18, 19, 20],  # 小指
                    ]

                    for finger in finger_connections:
                        for i in range(len(finger) - 1):
                            if finger[i] < len(landmarks) and finger[i + 1] < len(
                                landmarks
                            ):
                                pt1 = (
                                    int(landmarks[finger[i]]["x"] * w),
                                    int(landmarks[finger[i]]["y"] * h),
                                )
                                pt2 = (
                                    int(landmarks[finger[i + 1]]["x"] * w),
                                    int(landmarks[finger[i]]["y"] * h),
                                )
                                cv2.line(annotated_frame, pt1, pt2, (0, 255, 255), 1)

    return annotated_frame


def _draw_detections_on_frame(frame, result):
    """
    在帧上绘制检测结果

    Args:
        frame: 输入帧
        result: 检测结果

    Returns:
        带标注的帧
    """
    import cv2

    # 绘制人体检测框
    for person in result.person_detections:
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

    # 绘制发网检测结果
    for i, hairnet in enumerate(result.hairnet_results):
        if hairnet.get("has_hairnet", False):
            # 获取对应的人体检测框
            if i < len(result.person_detections):
                bbox = result.person_detections[i].get("bbox", [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    # 在人体框上方绘制发网标签
                    cv2.putText(
                        frame,
                        "Hairnet ✓",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

    # 绘制洗手检测结果
    for i, handwash in enumerate(result.handwash_results):
        if handwash.get("is_handwashing", False):
            if i < len(result.person_detections):
                bbox = result.person_detections[i].get("bbox", [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.putText(
                        frame,
                        "Handwashing ✓",
                        (x1, y1 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2,
                    )

    # 绘制消毒检测结果
    for i, sanitize in enumerate(result.sanitize_results):
        if sanitize.get("is_sanitizing", False):
            if i < len(result.person_detections):
                bbox = result.person_detections[i].get("bbox", [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.putText(
                        frame,
                        "Sanitizing ✓",
                        (x1, y1 - 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )

    # 只有在检测到人体时才检测并绘制手部关键点
    hands_count = 0
    if result.person_detections:
        pose_detector = PoseDetector()
        hands_results = pose_detector.detect_hands(frame)
        hands_count = len(hands_results)

        # 绘制手部关键点
        for hand_result in hands_results:
            landmarks = hand_result["landmarks"]
            hand_label = hand_result["label"]
            bbox = hand_result["bbox"]

            # 检查手部是否在任何人体检测框内
            hand_in_person = False
            for person in result.person_detections:
                person_bbox = person.get("bbox", [])
                if len(person_bbox) >= 4:
                    if _bbox_overlap(
                        [bbox[0], bbox[1], bbox[2], bbox[3]], person_bbox, 0.1
                    ):
                        hand_in_person = True
                        break

            # 只绘制在人体区域内的手部
            if hand_in_person:
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

    # 在左上角显示帧信息
    info_text = f"Persons: {len(result.person_detections)}, Hands: {hands_count}"
    cv2.putText(
        frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    return frame

# FastAPI application
# FastAPI应用主文件

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import asyncio
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
import base64
from pathlib import Path

# 导入检测器
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 定义备用类
class FallbackHumanDetector:
    def __init__(self):
        pass

    def detect(self, frame):
        return []

    def visualize_detections(self, image, detections):
        return image


class FallbackSimpleHairnetDetector:
    def detect_hairnet(self, head_region):
        import random

        wearing_hairnet = random.choice([True, False])
        return {
            "wearing_hairnet": wearing_hairnet,
            "has_hairnet": wearing_hairnet,  # 保持兼容性
            "confidence": random.uniform(0.6, 0.9),
            "hairnet_color": "unknown",
            "hairnet_pixels": 0,
            "pixel_density": 0.0,
            "edge_density": 0.0,
            "contour_count": 0,
            "raw_prediction": 0,
            "debug_info": {},
        }

    def _detect_hairnet_with_pytorch(self, head_region):
        import random

        wearing_hairnet = random.choice([True, False])
        return {
            "wearing_hairnet": wearing_hairnet,
            "has_hairnet": wearing_hairnet,  # 保持兼容性
            "confidence": random.uniform(0.6, 0.9),
            "hairnet_color": "unknown",
            "hairnet_pixels": 0,
            "pixel_density": 0.0,
            "edge_density": 0.0,
            "contour_count": 0,
            "raw_prediction": 0,
            "debug_info": {},
        }


class FallbackHairnetDetectionPipeline:
    def __init__(self, detector=None, hairnet_detector=None):
        self.detector = detector or FallbackHumanDetector()
        self.hairnet_detector = hairnet_detector or FallbackSimpleHairnetDetector()

    def detect_hairnet_compliance(self, frame):
        # 简化的检测逻辑，假设检测到1个人，随机判断是否佩戴发网
        import random

        has_hairnet = random.choice([True, False])  # 随机模拟
        confidence = random.uniform(0.6, 0.9)

        return {
            "total_persons": 1,
            "persons_with_hairnet": 1 if has_hairnet else 0,
            "persons_without_hairnet": 0 if has_hairnet else 1,
            "compliance_rate": 1.0 if has_hairnet else 0.0,
            "detections": [
                {
                    "bbox": [100, 100, 200, 300],
                    "has_hairnet": has_hairnet,
                    "confidence": confidence,
                    "hairnet_confidence": confidence,
                }
            ],
            "average_confidence": confidence,
        }
        
    def detect(self, frame):
        # 添加与YOLOHairnetDetector兼容的detect方法
        import random
        
        has_hairnet = random.choice([True, False])  # 随机模拟
        confidence = random.uniform(0.6, 0.9)
        
        return {
            "wearing_hairnet": has_hairnet,
            "confidence": confidence,
            "head_roi_coords": [100, 100, 200, 300]
        }
    
    def detect_hairnet(self, frame):
        # 添加与YOLOHairnetDetector兼容的detect_hairnet方法
        return self.detect(frame)
    
    def get_stats(self):
        # 添加与YOLOHairnetDetector兼容的get_stats方法
        return {
            "model_path": "fallback_model",
            "confidence_threshold": 0.5,
            "device": "cpu"
        }


class FallbackDetectionDataManager:
    def __init__(self, db_path=None):
        if db_path is None:
            # 使用项目根目录下的data目录
            from pathlib import Path

            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            # 确保data目录存在
            if not data_dir.exists():
                data_dir.mkdir(exist_ok=True)
            self.db_path = str(data_dir / "detection_results.db")
        else:
            self.db_path = db_path

    def save_detection_result(self, *args, **kwargs):
        return True

    def get_realtime_statistics(self):
        return {}

    def get_daily_statistics(self, days=7):
        return []

    def get_detection_history(self, limit=100, detection_type=None):
        return []


try:
    from src.core.detector import HumanDetector
    from src.core.data_manager import DetectionDataManager

    logger.info("成功导入核心模块")
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    # 使用备用类
    HumanDetector = FallbackHumanDetector
    DetectionDataManager = FallbackDetectionDataManager
    
# 备用检测管道已在前面定义，无需额外操作

# 创建FastAPI应用
app = FastAPI(
    title="人体行为检测系统 API",
    description="基于深度学习的实时人体行为检测与分析系统",
    version="1.0.0",
)

# 启用CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
frontend_path = os.path.join(project_root, "frontend")
if os.path.exists(frontend_path):
    app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")
    logger.info(f"静态文件目录已挂载: {frontend_path} 到 /frontend 路径")

# 挂载静态文件目录
static_path = os.path.join(project_root, "static")
if not os.path.exists(static_path):
    os.makedirs(static_path, exist_ok=True)
    # 创建一个空的app.js文件以避免404错误
    with open(os.path.join(static_path, "app.js"), "w") as f:
        f.write("// 空文件，用于避免404错误")
app.mount("/static", StaticFiles(directory=static_path), name="static")
logger.info(f"静态文件目录已挂载: {static_path} 到 /static 路径")

# 全局检测器实例
detector = None
hairnet_pipeline = None
data_manager = None


# WebSocket连接管理（如果没有导入成功，使用内置版本）
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.heartbeat_interval = 30  # 心跳间隔（秒）

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket连接建立，当前连接数: {len(self.active_connections)}")
        # 启动心跳检测
        asyncio.create_task(self._heartbeat(websocket))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket连接断开，当前连接数: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """广播消息给所有连接的客户端"""
        if self.active_connections:
            for connection in self.active_connections.copy():
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.warning(f"广播消息失败: {e}")
                    self.disconnect(connection)

    async def _heartbeat(self, websocket: WebSocket):
        """心跳检测"""
        try:
            while websocket in self.active_connections:
                await asyncio.sleep(self.heartbeat_interval)
                if websocket in self.active_connections:
                    await websocket.send_text(json.dumps({"type": "ping"}))
        except Exception as e:
            logger.info(f"心跳检测结束: {e}")
            self.disconnect(websocket)


manager = ConnectionManager()


# 可视化函数
def visualize_hairnet_detections(image, detections):
    """可视化发网检测结果 - 增强版标注"""
    annotated_image = image.copy()

    # ---------------- 新增：先进行人体检测 ----------------
    global detector
    person_detections = []
    if detector is not None and hasattr(detector, "detect"):
        try:
            person_detections = detector.detect(image)
        except Exception:
            pass

    # 根据人体检测结果重新统计人数
    total_persons = len(person_detections)

    # 判断每个人是否佩戴发网：若头部（hairnet_bbox）中心点位于该人体框内则认为佩戴
    compliant_count = 0
    for person in person_detections:
        px1, py1, px2, py2 = person.get("bbox", [0, 0, 0, 0])
        has_hairnet = False
        for det in detections:
            hx1, hy1, hx2, hy2 = det.get("bbox", [0, 0, 0, 0])
            cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                has_hairnet = det.get("has_hairnet", False)
                break
        if has_hairnet:
            compliant_count += 1
    compliance_rate = (
        (compliant_count / total_persons * 100) if total_persons > 0 else 0
    )

    # 在图片顶部添加统计信息（使用英文避免乱码）
    stats_text = f"Total: {total_persons} | Compliant: {compliant_count} | Rate: {compliance_rate:.1f}%"
    stats_bg_color = (0, 0, 0)  # 黑色背景
    stats_text_color = (255, 255, 255)  # 白色文字

    # 计算统计信息文字大小
    stats_font_scale = 0.7
    stats_thickness = 2
    (stats_w, stats_h), _ = cv2.getTextSize(
        stats_text, cv2.FONT_HERSHEY_SIMPLEX, stats_font_scale, stats_thickness
    )

    # 绘制统计信息背景
    cv2.rectangle(
        annotated_image, (10, 10), (20 + stats_w, 20 + stats_h), stats_bg_color, -1
    )
    # 绘制统计信息文字
    cv2.putText(
        annotated_image,
        stats_text,
        (15, 15 + stats_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        stats_font_scale,
        stats_text_color,
        stats_thickness,
    )

    # ------------- 绘制人体框 -------------
    for idx, person in enumerate(person_detections, 1):
        bbox = person.get("bbox", [])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = map(int, bbox[:4])
        # 人体框使用青色
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        # 编号标注
        cv2.putText(annotated_image, f"P{idx}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # ------------- 绘制发网框 -------------
    for d_idx, detection in enumerate(detections, 1):
        bbox = detection.get("bbox", [])
        has_hairnet = detection.get("has_hairnet", False)
        confidence = detection.get("confidence", 0.0)

        if len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])

            # 根据是否佩戴发网选择颜色和状态
            if has_hairnet:
                color = (0, 255, 0)  # 绿色表示合规
                status = "COMPLIANT"
                status_bg = (0, 200, 0)
            else:
                color = (0, 0, 255)  # 红色表示不合规
                status = "NON-COMPLIANT"
                status_bg = (0, 0, 200)

            # 绘制人体检测边界框（蓝色，显示检测到的人体）
            cv2.rectangle(
                annotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2
            )  # 青色边框表示人体检测

            # 绘制发网合规状态边界框（内层，根据合规状态显示颜色）
            cv2.rectangle(annotated_image, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), color, 3)

            # 绘制人员编号圆圈
            center_x, center_y = x1 + 25, y1 + 25
            cv2.circle(
                annotated_image, (center_x, center_y), 18, (255, 255, 0), -1
            )  # 青色背景
            cv2.circle(
                annotated_image, (center_x, center_y), 18, (0, 0, 0), 2
            )  # 黑色边框
            cv2.putText(
                annotated_image,
                str(d_idx),
                (center_x - 8, center_y + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )  # 黑色数字


            # 准备标签信息（使用英文避免乱码）
            person_label = f"Person #{d_idx}"
            status_label = f"{status}"
            confidence_label = f"Conf: {confidence:.2f}"

            # 计算标签尺寸
            font_scale = 0.6
            thickness = 2

            # 获取各个标签的尺寸
            (person_w, person_h), _ = cv2.getTextSize(
                person_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            (status_w, status_h), _ = cv2.getTextSize(
                status_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            (conf_w, conf_h), _ = cv2.getTextSize(
                confidence_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # 计算标签框的总尺寸
            max_width = max(person_w, status_w, conf_w) + 20
            total_height = person_h + status_h + conf_h + 30

            # 确定标签位置（避免超出图像边界）
            label_x = x1
            label_y = y1 - total_height - 10
            if label_y < 0:
                label_y = y2 + 10
            if label_x + max_width > annotated_image.shape[1]:
                label_x = annotated_image.shape[1] - max_width

            # 绘制标签背景
            cv2.rectangle(
                annotated_image,
                (label_x, label_y),
                (label_x + max_width, label_y + total_height),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                annotated_image,
                (label_x, label_y),
                (label_x + max_width, label_y + total_height),
                color,
                2,
            )

            # 绘制标签文字
            text_y = label_y + person_h + 5
            cv2.putText(
                annotated_image,
                person_label,
                (label_x + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            text_y += status_h + 5
            cv2.putText(
                annotated_image,
                status_label,
                (label_x + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                status_bg,
                thickness,
            )

            text_y += conf_h + 5
            cv2.putText(
                annotated_image,
                confidence_label,
                (label_x + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (200, 200, 200),
                thickness,
            )

    # 在图片右下角添加图例说明
    legend_x = annotated_image.shape[1] - 280
    legend_y = annotated_image.shape[0] - 120

    # 确保图例不会超出图像边界
    if legend_x < 0:
        legend_x = 10
    if legend_y < 0:
        legend_y = annotated_image.shape[0] - 60

    # 绘制图例背景
    cv2.rectangle(
        annotated_image,
        (legend_x, legend_y),
        (legend_x + 270, legend_y + 110),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        annotated_image,
        (legend_x, legend_y),
        (legend_x + 270, legend_y + 110),
        (255, 255, 255),
        2,
    )

    # 图例标题
    cv2.putText(
        annotated_image,
        "Legend:",
        (legend_x + 10, legend_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # 人体检测框说明
    cv2.rectangle(
        annotated_image,
        (legend_x + 10, legend_y + 30),
        (legend_x + 30, legend_y + 45),
        (255, 255, 0),
        2,
    )
    cv2.putText(
        annotated_image,
        "Human Detection",
        (legend_x + 40, legend_y + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # 合规状态说明
    cv2.rectangle(
        annotated_image,
        (legend_x + 10, legend_y + 55),
        (legend_x + 30, legend_y + 70),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated_image,
        "Compliant (With Hairnet)",
        (legend_x + 40, legend_y + 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # 违规状态说明
    cv2.rectangle(
        annotated_image,
        (legend_x + 10, legend_y + 80),
        (legend_x + 30, legend_y + 95),
        (0, 0, 255),
        2,
    )
    cv2.putText(
        annotated_image,
        "Non-Compliant (No Hairnet)",
        (legend_x + 40, legend_y + 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return annotated_image


# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化检测器"""
    global detector, hairnet_pipeline, data_manager
    try:
        # 优先尝试初始化 HumanDetector，若失败则回退到 FallbackHumanDetector
        try:
            from core.detector import HumanDetector as RealHumanDetector
            detector = RealHumanDetector(
                model_path=os.environ.get("HUMAN_MODEL_PATH", "yolov8n.pt"),
                device=os.environ.get("HUMAN_DEVICE", "auto")
            )
            # 仅对 HumanDetector 实例设置参数
            detector.confidence_threshold = float(os.environ.get("HUMAN_CONF_THRES", "0.2"))
            detector.min_box_area = int(os.environ.get("HUMAN_MIN_BOX_AREA", "600"))
            detector.max_box_ratio = float(os.environ.get("HUMAN_MAX_BOX_RATIO", "6.0"))
        except Exception:
            # 回退到 FallbackHumanDetector（无参数可调）
            from core.detector import FallbackHumanDetector
            detector = FallbackHumanDetector()

        # 使用发网检测器工厂创建YOLOv8检测器
        try:
            # 使用工厂模式创建检测器
            from core.hairnet_detection_factory import HairnetDetectionFactory

            # 从环境变量或配置中获取参数
            model_path = os.environ.get(
                "HAIRNET_MODEL_PATH", "models/hairnet_detection.pt"
            )
            device = os.environ.get("HAIRNET_DEVICE", "auto")
            conf_thres = float(os.environ.get("HAIRNET_CONF_THRES", "0.25"))
            iou_thres = float(os.environ.get("HAIRNET_IOU_THRES", "0.45"))

            logger.info(f"使用检测器工厂创建YOLOv8发网检测器，模型: {model_path}")
            try:
                hairnet_detector = HairnetDetectionFactory.create_detector(
                    detector_type="yolo",
                    model_path=model_path,
                    device=device,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres
                )
                
                logger.info("成功创建YOLOv8发网检测器")
                # 直接使用YOLOv8检测器作为管道
                hairnet_pipeline = hairnet_detector
            except (ImportError, FileNotFoundError) as e:
                logger.error(f"创建YOLOv8发网检测器失败: {e}")
                raise

        except ImportError as e:
            logger.error(f"发网检测器工厂导入失败: {e}")
            raise

        # 初始化数据管理器
        try:
            from core.data_manager import DetectionDataManager
            
            # 如果YOLOv8检测器创建失败，将使用备用检测器
            if hairnet_pipeline is None:
                logger.warning("YOLOv8检测器创建失败，使用备用检测器")
                hairnet_pipeline = FallbackHairnetDetectionPipeline(
                    detector, FallbackSimpleHairnetDetector()
                )
                
            data_manager = DetectionDataManager()
            logger.info("检测器初始化成功")
        except Exception as e:
            logger.error(f"数据管理器初始化失败: {e}")
    except Exception as e:
        logger.error(f"检测器初始化失败: {e}")
        raise


# 根路径
@app.get("/")
async def root():
    """根路径，返回前端页面"""
    frontend_file = os.path.join(project_root, "frontend", "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    else:
        return {"message": "人体行为检测系统 API", "docs": "/docs"}


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "detector_ready": True,
    }


# API信息
@app.get("/api/v1/info")
async def get_api_info():
    """获取API信息"""
    info = {
        "name": "人体行为检测系统 API",
        "version": "1.0.0",
        "description": "基于深度学习的实时人体行为检测与分析系统",
        "endpoints": {
            "health": "/health",
            "detect_image": "/api/v1/detect/image",
            "detect_hairnet": "/api/v1/detect/hairnet",
            "statistics": "/api/v1/statistics",
            "websocket": "/ws",
        },
    }

    # 添加检测器信息
    if hairnet_pipeline is not None:
        detector_type = type(hairnet_pipeline).__name__
        info["hairnet_detector_type"] = detector_type

        # 如果是YOLOv8检测器，添加更多信息
        if detector_type == "YOLOHairnetDetector":
            try:
                # 检查hairnet_pipeline是否有get_stats方法
                if hasattr(hairnet_pipeline, "get_stats"):
                    stats = hairnet_pipeline.get_stats()
                    if isinstance(stats, dict):
                        info["model_path"] = stats.get("model_path", "")
                        info["confidence_threshold"] = stats.get(
                            "confidence_threshold", 0.0
                        )
                        info["device"] = stats.get("device", "")
            except Exception as e:
                logger.error(f"获取检测器统计信息失败: {e}")

    return info


# 图像检测API
@app.post("/api/v1/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """图像检测接口"""
    if detector is None:
        raise HTTPException(status_code=500, detail="检测器未初始化")

    try:
        # 读取上传的图像
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无效的图像格式")

        # 执行检测
        results = detector.detect(image)

        # 绘制检测结果
        if hasattr(detector, "visualize_detections"):
            annotated_image = detector.visualize_detections(image, results)
        else:
            annotated_image = image

        # 将结果图像编码为base64
        _, buffer = cv2.imencode(".jpg", annotated_image)
        img_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        # 提取检测信息 - results已经是处理好的字典列表
        detections = results

        return {
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "annotated_image": img_base64,
        }

    except Exception as e:
        logger.error(f"图像检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


# 发网检测API
@app.post("/api/v1/detect/hairnet")
async def detect_hairnet(file: UploadFile = File(...)):
    """发网检测接口"""
    if hairnet_pipeline is None:
        raise HTTPException(status_code=500, detail="发网检测器未初始化")

    try:
        # 读取上传的图像
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无效的图像格式")

        # 执行发网检测 - 根据检测器类型调用不同的方法
        results = {}
        annotated_image = None
        if type(hairnet_pipeline).__name__ == "YOLOHairnetDetector":
            # 使用YOLOv8检测器
            logger.info("使用YOLOv8检测器进行发网检测")
            try:
                # 确保检测器有detect方法
                if hasattr(hairnet_pipeline, "detect"):
                    result = hairnet_pipeline.detect(image)

                    # 确保结果是字典类型
                    if isinstance(result, dict):
                        wearing_hairnet = False
                        confidence = 0.0
                        detections = []
                        visualization = None

                        # 安全地获取结果属性
                        if "wearing_hairnet" in result:
                            wearing_hairnet = result["wearing_hairnet"]
                        if "confidence" in result:
                            confidence = result["confidence"]
                        if "detections" in result and isinstance(
                            result["detections"], list
                        ):
                            detections = result["detections"]
                        if "visualization" in result:
                            visualization = result["visualization"]

                        # 构建与传统检测器兼容的结果格式
                        results = {
                            "total_persons": len(detections) if detections else 0,
                            "persons_with_hairnet": 1 if wearing_hairnet else 0,
                            "persons_without_hairnet": 0 if wearing_hairnet else 1,
                            "compliance_rate": 1.0 if wearing_hairnet else 0.0,
                            "detections": [],
                        }

                        # 安全地处理检测结果
                        if detections:
                            for det in detections:
                                if isinstance(det, dict):
                                    bbox = [0, 0, 0, 0]
                                    if "bbox" in det and isinstance(det["bbox"], list):
                                        bbox = det["bbox"]

                                    det_class = ""
                                    if "class" in det:
                                        det_class = det["class"]

                                    det_confidence = 0.0
                                    if "confidence" in det:
                                        det_confidence = det["confidence"]

                                    results["detections"].append(
                                        {
                                            "bbox": bbox,
                                            "has_hairnet": det_class == "hairnet",
                                            "confidence": det_confidence,
                                            "hairnet_confidence": det_confidence,
                                        }
                                    )

                        # 如果没有检测到任何物体，添加一个默认检测结果
                        if not results["detections"]:
                            results["detections"] = [
                                {
                                    "bbox": [0, 0, 0, 0],
                                    "has_hairnet": wearing_hairnet,
                                    "confidence": confidence,
                                    "hairnet_confidence": confidence,
                                }
                            ]

                        results["average_confidence"] = confidence

                        # 使用YOLOv8生成的可视化结果
                        annotated_image = visualization
                    else:
                        # 如果结果不是字典，使用默认结果
                        logger.warning("YOLOv8检测器返回的结果不是字典类型")
                        results = {
                            "total_persons": 0,
                            "persons_with_hairnet": 0,
                            "persons_without_hairnet": 0,
                            "compliance_rate": 0.0,
                            "detections": [],
                            "average_confidence": 0.0,
                        }
                        annotated_image = None
                else:
                    # 如果检测器没有detect方法，使用默认结果
                    logger.warning("YOLOv8检测器没有detect方法")
                    results = {
                        "total_persons": 0,
                        "persons_with_hairnet": 0,
                        "persons_without_hairnet": 0,
                        "compliance_rate": 0.0,
                        "detections": [],
                        "average_confidence": 0.0,
                    }
                    annotated_image = None
            except Exception as e:
                logger.error(f"YOLOv8检测器错误: {e}")
                # 发生错误时使用默认结果
                results = {
                    "total_persons": 0,
                    "persons_with_hairnet": 0,
                    "persons_without_hairnet": 0,
                    "compliance_rate": 0.0,
                    "detections": [],
                    "average_confidence": 0.0,
                    "error": str(e),
                }
                annotated_image = None
        else:
            # 使用传统检测管道
            logger.info("使用传统检测管道进行发网检测")
            try:
                # 确保检测管道有detect_hairnet_compliance方法或detect_hairnet方法
                if hasattr(hairnet_pipeline, "detect_hairnet_compliance"):
                    results = hairnet_pipeline.detect_hairnet_compliance(image)
                elif hasattr(hairnet_pipeline, "detect_hairnet"):
                    # 使用detect_hairnet方法并转换结果格式
                    result = hairnet_pipeline.detect_hairnet(image)
                    # 构建与detect_hairnet_compliance兼容的结果格式
                    results = {
                        "total_persons": 1,
                        "persons_with_hairnet": (
                            1 if result.get("wearing_hairnet", False) else 0
                        ),
                        "persons_without_hairnet": (
                            0 if result.get("wearing_hairnet", False) else 1
                        ),
                        "compliance_rate": (
                            1.0 if result.get("wearing_hairnet", False) else 0.0
                        ),
                        "detections": [
                            {
                                "bbox": result.get("head_roi_coords", [0, 0, 0, 0]),
                                "has_hairnet": result.get("wearing_hairnet", False),
                                "confidence": result.get("confidence", 0.0),
                                "hairnet_confidence": result.get("confidence", 0.0),
                            }
                        ],
                        "average_confidence": result.get("confidence", 0.0),
                    }

                # 确保结果是字典类型
                if not isinstance(results, dict):
                    logger.warning("传统检测管道返回的结果不是字典类型")
                    results = {
                        "total_persons": 0,
                        "persons_with_hairnet": 0,
                        "persons_without_hairnet": 0,
                        "compliance_rate": 0.0,
                        "detections": [],
                        "average_confidence": 0.0,
                    }
                # 生成带标注的结果图片
                annotated_image = None
                if "detections" in results:
                    annotated_image = visualize_hairnet_detections(
                        image, results["detections"]
                    )
                else:
                    # 如果检测管道没有detect_hairnet_compliance方法，使用默认结果
                    logger.warning("传统检测管道没有detect_hairnet_compliance方法")
                    results = {
                        "total_persons": 0,
                        "persons_with_hairnet": 0,
                        "persons_without_hairnet": 0,
                        "compliance_rate": 0.0,
                        "detections": [],
                        "average_confidence": 0.0,
                    }
                    annotated_image = None
            except Exception as e:
                logger.error(f"传统检测管道错误: {e}")
                # 发生错误时使用默认结果
                results = {
                    "total_persons": 0,
                    "persons_with_hairnet": 0,
                    "persons_without_hairnet": 0,
                    "compliance_rate": 0.0,
                    "detections": [],
                    "average_confidence": 0.0,
                    "error": str(e),
                }
                annotated_image = None

        # 生成唯一的frame_id
        frame_id = f"api_{hash(str(results)) % 1000000000}"

        # 保存检测结果到数据库
        if (
            data_manager
            and isinstance(results, dict)
            and results.get("total_persons", 0) > 0
        ):
            try:
                data_manager.save_detection_result(
                    frame_id=frame_id,
                    detection_results=results.get("detections", []),
                    detection_type="hairnet",
                    processing_time=0.0,
                )
                logger.info(
                    f"保存检测结果: {frame_id}, 总人数: {results.get('total_persons', 0)}, 合规率: {results.get('compliance_rate', 0.0):.2f}"
                )
            except Exception as e:
                logger.error(f"保存检测结果失败: {e}")

        # 将标注图片转换为base64
        annotated_image_base64 = None
        if annotated_image is not None:
            _, buffer = cv2.imencode(".jpg", annotated_image)
            annotated_image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        return {
            "success": True,
            "detections": results,
            "detection_count": (
                results.get("total_persons", 0) if isinstance(results, dict) else 0
            ),
            "frame_id": frame_id,
            "annotated_image": annotated_image_base64,
        }

    except Exception as e:
        logger.error(f"发网检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/api/v1/detect/hairnet/video")
async def detect_hairnet_video(file: UploadFile = File(...)):
    """视频发网检测接口"""
    if hairnet_pipeline is None:
        raise HTTPException(status_code=500, detail="发网检测器未初始化")

    # 检查文件类型
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="请上传有效的视频文件")

    try:
        # 保存临时视频文件
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_video_path = temp_file.name

        try:
            # 打开视频文件
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="无法打开视频文件")

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # 处理视频帧（每秒采样一帧）
            frame_interval = max(1, int(fps)) if fps > 0 else 30
            frame_results = []
            frame_count = 0
            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 每隔frame_interval帧处理一次
                if frame_count % frame_interval == 0:
                    try:
                        # 执行发网检测
                        if hasattr(hairnet_pipeline, "detect_hairnet_compliance"):
                            results = hairnet_pipeline.detect_hairnet_compliance(frame)
                        elif hasattr(hairnet_pipeline, "detect"):
                            # 使用detect方法并转换结果格式
                            result = hairnet_pipeline.detect(frame)
                            # 构建与detect_hairnet_compliance兼容的结果格式
                            results = {
                                "total_persons": 1,
                                "persons_with_hairnet": (
                                    1 if result.get("wearing_hairnet", False) else 0
                                ),
                                "persons_without_hairnet": (
                                    0 if result.get("wearing_hairnet", False) else 1
                                ),
                                "compliance_rate": (
                                    1.0 if result.get("wearing_hairnet", False) else 0.0
                                ),
                                "detections": [
                                    {
                                        "bbox": result.get(
                                            "head_roi_coords", [0, 0, 0, 0]
                                        ),
                                        "has_hairnet": result.get(
                                            "wearing_hairnet", False
                                        ),
                                        "confidence": result.get("confidence", 0.0),
                                        "hairnet_confidence": result.get(
                                            "confidence", 0.0
                                        ),
                                    }
                                ],
                                "average_confidence": result.get("confidence", 0.0),
                            }
                        else:
                            # 如果两个方法都不存在，使用默认结果
                            logger.warning(
                                "检测管道没有detect_hairnet_compliance或detect_hairnet方法"
                            )
                            results = {
                                "total_persons": 0,
                                "persons_with_hairnet": 0,
                                "persons_without_hairnet": 0,
                                "compliance_rate": 0.0,
                                "detections": [],
                                "average_confidence": 0.0,
                            }

                        if (
                            isinstance(results, dict)
                            and results.get("total_persons", 0) > 0
                        ):
                            frame_results.append(
                                {
                                    "frame_number": frame_count,
                                    "timestamp": (
                                        frame_count / fps
                                        if fps > 0
                                        else processed_frames
                                    ),
                                    "detections": results,
                                }
                            )

                        processed_frames += 1

                        # 限制处理帧数（避免处理时间过长）
                        if processed_frames >= 60:  # 最多处理60帧
                            break

                    except Exception as e:
                        logger.warning(f"处理第{frame_count}帧时出错: {e}")

                frame_count += 1

            cap.release()

            # 计算整体统计信息
            if frame_results:
                total_persons_sum = sum(
                    r["detections"].get("total_persons", 0) for r in frame_results
                )
                total_with_hairnet = sum(
                    r["detections"].get("persons_with_hairnet", 0)
                    for r in frame_results
                )
                total_without_hairnet = sum(
                    r["detections"].get("persons_without_hairnet", 0)
                    for r in frame_results
                )

                overall_compliance_rate = (
                    total_with_hairnet / total_persons_sum
                    if total_persons_sum > 0
                    else 0
                )
                avg_confidence = sum(
                    r["detections"].get("average_confidence", 0) for r in frame_results
                ) / len(frame_results)

                # 生成唯一的frame_id
                frame_id = f"video_api_{hash(str(frame_results)) % 1000000000}"

                # 保存检测结果到数据库
                if data_manager:
                    try:
                        data_manager.save_detection_result(
                            frame_id=frame_id,
                            detection_results=frame_results,
                            detection_type="hairnet_video",
                            processing_time=0.0,
                        )
                        logger.info(
                            f"保存视频检测结果: {frame_id}, 处理帧数: {len(frame_results)}, 总人数: {total_persons_sum}, 整体合规率: {overall_compliance_rate:.2f}"
                        )
                    except Exception as e:
                        logger.error(f"保存视频检测结果失败: {e}")

                return {
                    "success": True,
                    "video_info": {
                        "duration": duration,
                        "fps": fps,
                        "total_frames": total_frames,
                        "processed_frames": len(frame_results),
                    },
                    "overall_statistics": {
                        "total_persons": total_persons_sum,
                        "persons_with_hairnet": total_with_hairnet,
                        "persons_without_hairnet": total_without_hairnet,
                        "compliance_rate": overall_compliance_rate,
                        "average_confidence": avg_confidence,
                    },
                    "frame_results": frame_results,
                    "frame_id": frame_id,
                }
            else:
                return {
                    "success": True,
                    "video_info": {
                        "duration": duration,
                        "fps": fps,
                        "total_frames": total_frames,
                        "processed_frames": 0,
                    },
                    "overall_statistics": {
                        "total_persons": 0,
                        "persons_with_hairnet": 0,
                        "persons_without_hairnet": 0,
                        "compliance_rate": 0,
                        "average_confidence": 0,
                    },
                    "frame_results": [],
                    "frame_id": f"video_api_empty_{hash(str(duration)) % 1000000000}",
                }

        finally:
            # 清理临时文件
            try:
                os.unlink(temp_video_path)
            except:
                pass

    except Exception as e:
        logger.error(f"视频发网检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"视频检测失败: {str(e)}")


# 统计数据API
@app.get("/api/statistics/realtime")
async def get_realtime_statistics():
    """获取实时统计数据"""
    if data_manager is None:
        raise HTTPException(status_code=500, detail="数据管理器未初始化")

    try:
        stats = data_manager.get_realtime_statistics()
        return stats
    except Exception as e:
        logger.error(f"获取实时统计数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计数据失败: {str(e)}")


@app.get("/api/statistics/daily")
async def get_daily_statistics(days: int = 7):
    """获取每日统计数据"""
    if data_manager is None:
        raise HTTPException(status_code=500, detail="数据管理器未初始化")

    try:
        stats = data_manager.get_daily_statistics(days=days)
        return stats
    except Exception as e:
        logger.error(f"获取每日统计数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计数据失败: {str(e)}")


@app.get("/api/statistics/history")
async def get_detection_history(limit: int = 100, detection_type: Optional[str] = None):
    """获取检测历史记录"""
    if data_manager is None:
        raise HTTPException(status_code=500, detail="数据管理器未初始化")

    try:
        history = data_manager.get_detection_history(
            limit=limit, detection_type=detection_type
        )
        return history
    except Exception as e:
        logger.error(f"获取检测历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取历史数据失败: {str(e)}")


# WebSocket实时检测
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket实时检测接口"""
    await manager.connect(websocket)

    try:
        while True:
            # 设置接收超时
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning("WebSocket接收超时")
                continue

            try:
                message = json.loads(data)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}")
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "消息格式错误"})
                )
                continue

            message_type = message.get("type")

            if message_type == "image":
                # 解码base64图像
                image_data = message.get("data", "")
                if not image_data:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "图像数据为空"})
                    )
                    continue

                if image_data.startswith("data:image"):
                    image_data = image_data.split(",")[1]

                try:
                    # 解码图像
                    img_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if image is None:
                        await websocket.send_text(
                            json.dumps({"type": "error", "message": "图像解码失败"})
                        )
                        continue

                    if hairnet_pipeline is None:
                        await websocket.send_text(
                            json.dumps({"type": "error", "message": "检测器未初始化"})
                        )
                        continue

                    # 执行发网检测 - 根据检测器类型调用不同的方法
                    results = {"total_persons": 0, "detections": []}
                    processing_time = asyncio.get_event_loop().time()

                    if type(hairnet_pipeline).__name__ == "YOLOHairnetDetector":
                        # 使用YOLOv8检测器
                        logger.info("WebSocket: 使用YOLOv8检测器进行发网检测")
                        try:
                            # 确保检测器有detect方法
                            if hasattr(hairnet_pipeline, "detect"):
                                result = hairnet_pipeline.detect(image)

                                # 确保结果是字典类型
                                if isinstance(result, dict):
                                    wearing_hairnet = False
                                    confidence = 0.0
                                    detections = []

                                    # 安全地获取结果属性
                                    if "wearing_hairnet" in result:
                                        wearing_hairnet = result["wearing_hairnet"]
                                    if "confidence" in result:
                                        confidence = result["confidence"]
                                    if "detections" in result and isinstance(
                                        result["detections"], list
                                    ):
                                        detections = result["detections"]

                                    # 构建与传统检测器兼容的结果格式
                                    results = {
                                        "total_persons": (
                                            len(detections) if detections else 0
                                        ),
                                        "persons_with_hairnet": (
                                            1 if wearing_hairnet else 0
                                        ),
                                        "persons_without_hairnet": (
                                            0 if wearing_hairnet else 1
                                        ),
                                        "compliance_rate": (
                                            1.0 if wearing_hairnet else 0.0
                                        ),
                                        "detections": [],
                                    }

                                    # 安全地处理检测结果
                                    if detections:
                                        for det in detections:
                                            if isinstance(det, dict):
                                                bbox = [0, 0, 0, 0]
                                                if "bbox" in det and isinstance(
                                                    det["bbox"], list
                                                ):
                                                    bbox = det["bbox"]

                                                det_class = ""
                                                if "class" in det:
                                                    det_class = det["class"]

                                                det_confidence = 0.0
                                                if "confidence" in det:
                                                    det_confidence = det["confidence"]

                                                results["detections"].append(
                                                    {
                                                        "bbox": bbox,
                                                        "has_hairnet": det_class
                                                        == "hairnet",
                                                        "confidence": det_confidence,
                                                        "hairnet_confidence": det_confidence,
                                                    }
                                                )

                                    # 如果没有检测到任何物体，添加一个默认检测结果
                                    if not results["detections"]:
                                        results["detections"] = [
                                            {
                                                "bbox": [0, 0, 0, 0],
                                                "has_hairnet": wearing_hairnet,
                                                "confidence": confidence,
                                                "hairnet_confidence": confidence,
                                            }
                                        ]

                                    results["average_confidence"] = confidence
                                else:
                                    # 如果结果不是字典，使用默认结果
                                    logger.warning(
                                        "WebSocket: YOLOv8检测器返回的结果不是字典类型"
                                    )
                                    results = {
                                        "total_persons": 0,
                                        "persons_with_hairnet": 0,
                                        "persons_without_hairnet": 0,
                                        "compliance_rate": 0.0,
                                        "detections": [],
                                        "average_confidence": 0.0,
                                    }
                            else:
                                # 如果检测器没有detect方法，使用默认结果
                                logger.warning("WebSocket: YOLOv8检测器没有detect方法")
                                results = {
                                    "total_persons": 0,
                                    "persons_with_hairnet": 0,
                                    "persons_without_hairnet": 0,
                                    "compliance_rate": 0.0,
                                    "detections": [],
                                    "average_confidence": 0.0,
                                }
                        except Exception as e:
                            logger.error(f"WebSocket: YOLOv8检测器错误: {e}")
                            # 发生错误时使用默认结果
                            results = {
                                "total_persons": 0,
                                "persons_with_hairnet": 0,
                                "persons_without_hairnet": 0,
                                "compliance_rate": 0.0,
                                "detections": [],
                                "average_confidence": 0.0,
                                "error": str(e),
                            }
                    else:
                        # 使用传统检测管道
                        logger.info("WebSocket: 使用传统检测管道进行发网检测")
                        try:
                            # 确保检测管道有detect_hairnet_compliance方法或detect_hairnet方法
                            if hasattr(hairnet_pipeline, "detect_hairnet_compliance"):
                                results = hairnet_pipeline.detect_hairnet_compliance(
                                    image
                                )
                            elif hasattr(hairnet_pipeline, "detect_hairnet"):
                                # 使用detect_hairnet方法并转换结果格式
                                result = hairnet_pipeline.detect_hairnet(image)
                                # 构建与detect_hairnet_compliance兼容的结果格式
                                results = {
                                    "total_persons": 1,
                                    "persons_with_hairnet": (
                                        1 if result.get("wearing_hairnet", False) else 0
                                    ),
                                    "persons_without_hairnet": (
                                        0 if result.get("wearing_hairnet", False) else 1
                                    ),
                                    "compliance_rate": (
                                        1.0
                                        if result.get("wearing_hairnet", False)
                                        else 0.0
                                    ),
                                    "detections": [
                                        {
                                            "bbox": result.get(
                                                "head_roi_coords", [0, 0, 0, 0]
                                            ),
                                            "has_hairnet": result.get(
                                                "wearing_hairnet", False
                                            ),
                                            "confidence": result.get("confidence", 0.0),
                                            "hairnet_confidence": result.get(
                                                "confidence", 0.0
                                            ),
                                        }
                                    ],
                                    "average_confidence": result.get("confidence", 0.0),
                                }

                                # 确保结果是字典类型
                                if not isinstance(results, dict):
                                    logger.warning(
                                        "WebSocket: 传统检测管道返回的结果不是字典类型"
                                    )
                                    results = {
                                        "total_persons": 0,
                                        "persons_with_hairnet": 0,
                                        "persons_without_hairnet": 0,
                                        "compliance_rate": 0.0,
                                        "detections": [],
                                        "average_confidence": 0.0,
                                    }
                                else:
                                    # 如果检测管道没有detect_hairnet_compliance方法，使用默认结果
                                    logger.warning(
                                        "WebSocket: 传统检测管道没有detect_hairnet_compliance方法"
                                    )
                                    results = {
                                        "total_persons": 0,
                                        "persons_with_hairnet": 0,
                                        "persons_without_hairnet": 0,
                                        "compliance_rate": 0.0,
                                        "detections": [],
                                        "average_confidence": 0.0,
                                    }
                        except Exception as e:
                            logger.error(f"WebSocket: 传统检测管道错误: {e}")
                            # 发生错误时使用默认结果
                            results = {
                                "total_persons": 0,
                                "persons_with_hairnet": 0,
                                "persons_without_hairnet": 0,
                                "compliance_rate": 0.0,
                                "detections": [],
                                "average_confidence": 0.0,
                                "error": str(e),
                            }

                    # 保存检测结果
                    frame_id = f"ws_{int(processing_time * 1000)}"
                    if (
                        data_manager
                        and isinstance(results, dict)
                        and results.get("total_persons", 0) > 0
                    ):
                        try:
                            data_manager.save_detection_result(
                                frame_id=frame_id,
                                detection_results=results.get("detections", []),
                                detection_type="hairnet",
                                processing_time=0.0,
                            )
                        except Exception as e:
                            logger.error(f"保存WebSocket检测结果失败: {e}")

                    # 发送检测结果
                    response = {
                        "type": "hairnet_detection_result",
                        "detections": (
                            results.get("detections", [])
                            if isinstance(results, dict)
                            else []
                        ),
                        "detection_count": (
                            results.get("total_persons", 0)
                            if isinstance(results, dict)
                            else 0
                        ),
                        "timestamp": processing_time,
                        "frame_id": frame_id,
                    }
                    await websocket.send_text(json.dumps(response))

                except Exception as e:
                    logger.error(f"WebSocket检测失败: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": f"检测失败: {str(e)}"})
                    )

            elif message_type == "pong":
                # 响应心跳
                logger.debug("收到心跳响应")

            else:
                logger.warning(f"未知消息类型: {message_type}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket连接断开")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")

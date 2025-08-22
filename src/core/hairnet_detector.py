import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.detector import HumanDetector

# 导入统一参数配置
try:
    from src.config.unified_params import get_unified_params
except ImportError:
    # 兼容性处理
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from src.config.unified_params import get_unified_params

logger = logging.getLogger(__name__)

# 导入增强的头部ROI提取器
try:
    # 尝试导入增强的头部ROI提取器
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.append(project_root)
    from scripts.improved_head_roi import ImprovedHeadROIExtractor

    ENHANCED_ROI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入增强ROI提取器: {e}，将使用基础方法")
    ENHANCED_ROI_AVAILABLE = False
    ImprovedHeadROIExtractor = None


class HairnetCNN(nn.Module):
    """发网检测的CNN模型"""

    def __init__(self, num_classes=2):
        super(HairnetCNN, self).__init__()

        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class HairnetDetector:
    """发网检测器

    结合人体检测和发网分类的完整检测流程
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        初始化发网检测器

        Args:
            model_path: 预训练模型路径
            device: 计算设备
        """
        # 获取统一参数配置
        self.params = get_unified_params().hairnet_detection

        # 使用统一配置或传入参数
        if model_path is None:
            model_path = self.params.model_path
        if device == "auto":
            device = self.params.device

        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

        # 使用统一参数配置
        self.confidence_threshold = self.params.confidence_threshold

        # 发网检测算法参数
        self.edge_density_threshold = self.params.edge_density_threshold
        self.sensitive_edge_threshold = self.params.sensitive_edge_threshold
        self.min_contour_count = self.params.min_contour_count
        self.light_blue_ratio_threshold = self.params.light_blue_ratio_threshold
        self.light_color_ratio_threshold = self.params.light_color_ratio_threshold
        self.white_ratio_threshold = self.params.white_ratio_threshold
        self.upper_region_ratio = self.params.upper_region_ratio
        self.upper_edge_density_threshold = self.params.upper_edge_density_threshold
        self.total_score_threshold = self.params.total_score_threshold
        self.confidence_boost_factor = self.params.confidence_boost_factor

        # 添加姿态检测器（可选）
        self.pose_detector = None
        self.human_detector = None

        # 初始化增强的头部ROI提取器
        if ENHANCED_ROI_AVAILABLE and ImprovedHeadROIExtractor is not None:
            try:
                self.enhanced_roi_extractor = ImprovedHeadROIExtractor()
                self.use_enhanced_roi = True
                logger.info("增强ROI提取器初始化成功")
            except Exception as e:
                logger.warning(f"增强ROI提取器初始化失败: {e}，使用基础方法")
                self.enhanced_roi_extractor = None
                self.use_enhanced_roi = False
        else:
            self.enhanced_roi_extractor = None
            self.use_enhanced_roi = False

        # 统计信息
        self.stats = {"keypoint_success": 0, "bbox_fallback": 0}
        self.total_detections = 0
        self.hairnet_detections = 0

        # 日志记录器
        self.logger = logger

    def get_stats(self) -> Dict[str, Any]:
        """获取检测器统计信息

        Returns:
            Dict[str, Any]: 包含检测统计信息的字典
        """
        compliance_rate = 0.0
        if self.total_detections > 0:
            compliance_rate = self.hairnet_detections / self.total_detections

        return {
            "total_detections": self.total_detections,
            "hairnet_detections": self.hairnet_detections,
            "compliance_rate": compliance_rate,
            "keypoint_success": self.stats.get("keypoint_success", 0),
            "bbox_fallback": self.stats.get("bbox_fallback", 0),
        }

        logger.info(f"HairnetDetector initialized on {self.device}")

    def _get_device(self, device: str) -> str:
        """获取计算设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self, model_path: Optional[str]):
        """加载发网检测模型"""
        model = HairnetCNN(num_classes=2)

        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint)
                logger.info(f"成功加载预训练模型: {model_path}")
            except Exception as e:
                logger.warning(f"模型加载失败: {e}，使用随机初始化模型")
        else:
            logger.info("使用随机初始化模型（演示模式）")

        model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        """获取图像预处理变换"""
        return transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _init_detectors(self):
        """初始化检测器（如果需要）"""
        if self.human_detector is None:
            try:
                self.human_detector = HumanDetector()
                self.logger.info("人体检测器初始化成功")
            except Exception as e:
                self.logger.error(f"人体检测器初始化失败: {e}")

    def detect_hairnet(
        self,
        frame: np.ndarray,
        human_bbox: Optional[List[int]] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """检测发网佩戴状态（支持传入关键点）"""
        try:
            # 检查输入图像是否有效
            if frame is None:
                return self._create_error_result("输入图像为空")

            if not isinstance(frame, np.ndarray):
                return self._create_error_result("输入图像格式无效")

            if frame.size == 0:
                return self._create_error_result("输入图像为空数组")

            # 检查依赖注入的检测器
            if self.human_detector is None:
                self.logger.warning("人体检测器未注入，使用内部初始化")
            self._init_detectors()

            # 如果没有提供人体边界框，先进行人体检测
            if human_bbox is None:
                if self.human_detector is None:
                    return self._create_error_result("人体检测器未初始化")
                human_results = self.human_detector.detect(frame)
                if not human_results:
                    return self._create_no_detection_result()
                human_bbox = human_results[0]["bbox"]

            # 多通道策略：尝试增强ROI、关键点ROI和BBox ROI
            roi_enhanced = None
            roi_kp = None
            roi_bbox = None

            # 首先尝试增强ROI提取（最优方法）
            if self.use_enhanced_roi:
                roi_enhanced = self._extract_head_roi_enhanced(
                    frame, human_bbox, keypoints
                )
                if roi_enhanced is not None:
                    self.logger.info(
                        f"增强ROI提取成功，方法: {roi_enhanced.get('strategy', 'unknown')}，质量评分: {roi_enhanced.get('quality_score', 0):.3f}"
                    )

            # 备选：关键点ROI
            if (
                roi_enhanced is None
                and keypoints is not None
                and self.pose_detector is not None
            ):
                roi_kp = self._optimize_head_roi_with_keypoints(
                    frame, human_bbox, keypoints
                )
                if roi_kp is not None:
                    self.logger.info("关键点ROI提取成功")

            # 最后备选：BBox ROI
            if roi_enhanced is None and roi_kp is None and human_bbox is not None:
                roi_bbox = self._extract_head_roi_from_bbox(frame, human_bbox)
                if roi_bbox is not None:
                    self.logger.info("BBox ROI提取成功")

            # 选择最佳ROI策略并更新统计
            head_roi = None
            head_roi_coords = None
            roi_strategy = None
            roi_quality_score = 0.0
            roi_method_info = None

            if roi_enhanced is not None:
                # 优先使用增强ROI（最佳方法）
                head_roi = roi_enhanced["roi"]
                head_roi_coords = roi_enhanced["coords"]
                roi_strategy = roi_enhanced["strategy"]
                roi_quality_score = roi_enhanced.get("quality_score", 0.0)
                roi_method_info = roi_enhanced.get("method_info", {})
                self.stats["keypoint_success"] += 1  # 增强方法通常包含关键点
                self.logger.info(f"使用增强头部ROI，策略: {roi_strategy}")
            elif roi_kp is not None and roi_bbox is not None:
                # 两个传统ROI都有效，优先使用关键点ROI（更精确）
                head_roi = roi_kp["roi"]
                head_roi_coords = roi_kp["coords"]
                roi_strategy = "keypoint"
                self.stats["keypoint_success"] += 1
                self.logger.info("使用关键点优化的头部ROI（传统方法）")
            elif roi_kp is not None:
                # 只有关键点ROI有效
                head_roi = roi_kp["roi"]
                head_roi_coords = roi_kp["coords"]
                roi_strategy = "keypoint"
                self.stats["keypoint_success"] += 1
                self.logger.info("使用关键点优化的头部ROI")
            elif roi_bbox is not None:
                # 只有BBox ROI有效
                head_roi = roi_bbox["roi"]
                head_roi_coords = roi_bbox["coords"]
                roi_strategy = "bbox"
                self.stats["bbox_fallback"] += 1
                self.logger.info("使用BBox比例的头部ROI")
            else:
                # 所有方法都失败
                self.logger.warning("所有ROI提取方法都失败")
                return self._create_no_detection_result()

            # 发网检测
            hairnet_result = self._simple_hairnet_detection(head_roi)

            # 计算关键点数量
            keypoint_count = 0
            if keypoints is not None:
                keypoint_count = len(keypoints)
                self.logger.info(f"检测到 {keypoint_count} 个关键点")

            # 添加头部ROI坐标信息和关键点数量
            hairnet_result["head_roi_coords"] = head_roi_coords
            hairnet_result["roi_strategy"] = roi_strategy
            hairnet_result["keypoint_count"] = keypoint_count
            hairnet_result["roi_quality_score"] = roi_quality_score

            # 添加增强ROI的详细信息
            if roi_method_info:
                hairnet_result["roi_method_info"] = roi_method_info
                hairnet_result["enhanced_roi_used"] = True
            else:
                hairnet_result["enhanced_roi_used"] = False

            # 添加调试信息
            self.logger.info(
                f"发网检测结果: 佩戴={hairnet_result['wearing_hairnet']}, "
                f"颜色={hairnet_result['hairnet_color']}, "
                f"置信度={hairnet_result['confidence']:.3f}, "
                f"像素数={hairnet_result['hairnet_pixels']}, "
                f"像素密度={hairnet_result.get('pixel_density', 0):.3f}"
            )

            # 更新统计
            self.total_detections += 1
            if hairnet_result["wearing_hairnet"]:
                self.hairnet_detections += 1

            return hairnet_result

        except Exception as e:
            self.logger.error(f"发网检测失败: {e}")
            return self._create_error_result(str(e))

    def _optimize_head_roi_with_keypoints(
        self, image: np.ndarray, person_bbox: Optional[List[int]], keypoints: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        使用关键点优化头部ROI提取

        Args:
            image: 输入图像
            person_bbox: 人体边界框 [x1, y1, x2, y2]，可选
            keypoints: 人体关键点数组

        Returns:
            包含ROI图像、坐标和策略的字典，失败时返回None
        """
        try:
            if keypoints is None or len(keypoints) == 0:
                return None

            # 提取头部相关关键点（假设使用COCO格式）
            # 0: 鼻子, 1: 左眼, 2: 右眼, 3: 左耳, 4: 右耳
            head_keypoints = keypoints[:5] if len(keypoints) >= 5 else keypoints

            # 过滤有效关键点（置信度 > 0.3）
            valid_points = []
            for i, kp in enumerate(head_keypoints):
                if len(kp) >= 3 and kp[2] > 0.3:  # x, y, confidence
                    valid_points.append([kp[0], kp[1]])

            if len(valid_points) < 2:
                self.logger.warning("有效头部关键点不足，无法优化ROI")
                return None

            # 计算头部边界框
            valid_points = np.array(valid_points)
            x_min, y_min = np.min(valid_points, axis=0)
            x_max, y_max = np.max(valid_points, axis=0)

            # 扩展边界框以包含完整头部
            width = x_max - x_min
            height = y_max - y_min

            # 向上扩展以包含头顶
            expand_top = height * 0.5
            expand_bottom = height * 0.2
            expand_sides = width * 0.3

            x1 = max(0, int(x_min - expand_sides))
            y1 = max(0, int(y_min - expand_top))
            x2 = min(image.shape[1], int(x_max + expand_sides))
            y2 = min(image.shape[0], int(y_max + expand_bottom))

            # 检查ROI有效性
            if x2 <= x1 or y2 <= y1:
                return None

            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            return {
                "roi": roi,
                "coords": [x1, y1, x2, y2],
                "strategy": "keypoint_optimized",
            }

        except Exception as e:
            self.logger.error(f"关键点ROI优化失败: {e}")
            return None

    def _extract_head_roi_enhanced(
        self,
        image: np.ndarray,
        person_bbox: Optional[List[int]],
        keypoints: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        使用增强算法提取头部ROI

        Args:
            image: 输入图像
            person_bbox: 人体边界框 [x1, y1, x2, y2]
            keypoints: 人体关键点（可选）

        Returns:
            包含ROI图像、坐标和策略的字典，失败时返回None
        """
        try:
            if not self.use_enhanced_roi or self.enhanced_roi_extractor is None:
                return None

            if person_bbox is None:
                return None

            # 转换关键点格式（如果有）
            kp_list = None
            if keypoints is not None and len(keypoints) > 0:
                # 将numpy数组转换为列表格式
                if isinstance(keypoints, np.ndarray):
                    if keypoints.ndim == 2:  # shape: (n, 3)
                        kp_list = keypoints.flatten().tolist()
                    else:  # shape: (n*3,)
                        kp_list = keypoints.tolist()
                else:
                    kp_list = keypoints

            # 转换bbox为float类型
            bbox_float = [float(x) for x in person_bbox]

            # 使用增强的多方法ROI提取
            (
                head_roi,
                roi_info,
            ) = self.enhanced_roi_extractor.extract_head_roi_multi_method(
                image, bbox_float, kp_list
            )

            if head_roi is None or roi_info is None:
                return None

            return {
                "roi": head_roi,
                "coords": roi_info.get("coords", person_bbox),
                "strategy": f"enhanced_{roi_info.get('best_method', 'unknown')}",
                "quality_score": roi_info.get("quality_score", 0.0),
                "method_info": roi_info,
            }

        except Exception as e:
            self.logger.error(f"增强ROI提取失败: {e}")
            return None

    def _extract_head_roi_from_bbox(
        self, image: np.ndarray, person_bbox: Optional[List[int]]
    ) -> Optional[Dict[str, Any]]:
        """
        从人体边界框提取头部ROI（备份方法）

        Args:
            image: 输入图像
            person_bbox: 人体边界框 [x1, y1, x2, y2]

        Returns:
            包含ROI图像、坐标和策略的字典，失败时返回None
        """
        try:
            if person_bbox is None:
                return None

            # 从人体边界框提取头部区域的简化实现
            x1, y1, x2, y2 = person_bbox
            person_height = y2 - y1
            head_height = int(person_height * 0.3)
            head_region = image[y1 : y1 + head_height, x1:x2]

            if head_region.size == 0:
                return None

            return {
                "roi": head_region,
                "coords": [x1, y1, x2, y1 + head_height],
                "strategy": "bbox_fallback",
            }

        except Exception as e:
            self.logger.error(f"BBox ROI提取失败: {e}")
            return None

    def _simple_hairnet_detection(self, head_region: np.ndarray) -> Dict[str, Any]:
        """
        在头部ROI中检测发网

        Args:
            head_region: 头部区域图像

        Returns:
            发网检测结果

        Raises:
            RuntimeError: 当 CNN 模型不可用时
        """
        if self.model is None:
            raise RuntimeError(
                "CNN 发网检测模型未加载。请检查：\n" "1. 模型文件是否存在\n" "2. 模型路径是否正确\n" "3. 模型是否正确初始化"
            )

        try:
            # 只使用CNN模型进行检测
            return self._detect_hairnet_with_cnn(head_region)
        except Exception as e:
            raise RuntimeError(f"CNN发网检测失败: {e}")

    def _detect_hairnet_with_cnn(self, head_region: np.ndarray) -> Dict[str, Any]:
        """
        使用CNN模型检测发网

        Args:
            head_region: 头部区域图像

        Returns:
            发网检测结果字典
        """
        try:
            # 检查模型是否可用
            if self.model is None:
                raise ValueError("CNN模型未加载")

            # 图像预处理
            # 转换为PIL图像以使用torchvision的transform
            if head_region.size == 0:
                raise ValueError("头部区域图像为空")

            # 转换BGR到RGB（OpenCV使用BGR，PIL使用RGB）
            rgb_image = cv2.cvtColor(head_region, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # 应用预处理变换
            input_tensor = self.transform(pil_image)
            # 添加批次维度
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            # 进行推理
            with torch.no_grad():
                outputs = self.model(input_batch)
                # 应用softmax获取概率
                probabilities = F.softmax(outputs, dim=1)
                # 获取预测类别和置信度
                predicted_class = torch.argmax(probabilities, dim=1).item()
                predicted_class_int = int(predicted_class)  # 确保是整数类型
                confidence = probabilities[0][predicted_class_int].item()

            # 分析颜色（使用现有方法）
            color_analysis = self._analyze_hairnet_color(head_region)

            # 构建结果字典
            has_hairnet = predicted_class == 1  # 假设类别1表示有发网

            # 如果置信度低于阈值，认为没有发网
            if confidence < self.confidence_threshold:
                has_hairnet = False

            # 计算发网像素（使用颜色分析）
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)

            # 检测浅蓝色范围
            light_blue_lower = np.array([90, 20, 80])
            light_blue_upper = np.array([140, 255, 255])
            light_blue_mask = cv2.inRange(hsv, light_blue_lower, light_blue_upper)

            # 检测其他浅色范围
            light_colors_lower = np.array([0, 0, 120])
            light_colors_upper = np.array([180, 120, 255])
            light_mask = cv2.inRange(hsv, light_colors_lower, light_colors_upper)

            # 检测白色和灰色发网
            white_lower = np.array([0, 0, 180])
            white_upper = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, white_lower, white_upper)

            # 检测绿色发网
            green_lower = np.array([40, 20, 80])
            green_upper = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)

            # 计算颜色比例
            light_blue_ratio = np.sum(light_blue_mask > 0) / (
                light_blue_mask.shape[0] * light_blue_mask.shape[1]
            )
            light_color_ratio = np.sum(light_mask > 0) / (
                light_mask.shape[0] * light_mask.shape[1]
            )
            white_ratio = np.sum(white_mask > 0) / (
                white_mask.shape[0] * white_mask.shape[1]
            )
            green_ratio = np.sum(green_mask > 0) / (
                green_mask.shape[0] * green_mask.shape[1]
            )
            total_color_ratio = (
                light_blue_ratio + light_color_ratio + white_ratio + green_ratio
            )

            # 构建结果
            result = {
                "wearing_hairnet": bool(has_hairnet),  # 统一使用wearing_hairnet键
                "has_hairnet": bool(has_hairnet),  # 保持向后兼容
                "confidence": float(confidence),  # 确保是Python float类型
                "raw_prediction": int(predicted_class),  # 原始预测类别
                "model_confidence": float(confidence),  # 模型置信度
                "hairnet_color": self._determine_hairnet_color(
                    float(light_blue_ratio),
                    float(light_color_ratio),
                    float(white_ratio),
                    float(green_ratio),
                ),
                "hairnet_pixels": int(
                    np.sum(light_blue_mask > 0)
                    + np.sum(light_mask > 0)
                    + np.sum(white_mask > 0)
                    + np.sum(green_mask > 0)
                ),
                "pixel_density": float(total_color_ratio),
                "detection_method": "cnn",  # 标记使用CNN方法
                "debug_info": {
                    "light_blue_ratio": float(light_blue_ratio),
                    "light_color_ratio": float(light_color_ratio),
                    "white_ratio": float(white_ratio),
                    "green_ratio": float(green_ratio),
                    "total_color_ratio": float(total_color_ratio),
                    "model_output": outputs.cpu().numpy().tolist()[0]
                    if isinstance(outputs, torch.Tensor)
                    else None,
                    "probabilities": probabilities.cpu().numpy().tolist()[0]
                    if isinstance(probabilities, torch.Tensor)
                    else None,
                },
            }

            return result

        except Exception as e:
            self.logger.error(f"CNN发网检测失败: {e}")
            # 返回一个错误结果，调用者会回退到传统方法
            raise

    def detect_hairnet_with_keypoints(
        self,
        frame: np.ndarray,
        human_bbox: Optional[List[int]] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """检测发网佩戴状态（支持传入关键点）"""
        try:
            # 检查依赖注入的检测器
            if self.human_detector is None:
                self.logger.warning("人体检测器未注入，使用内部初始化")

            # 如果没有提供人体边界框，先进行人体检测
            if human_bbox is None:
                if self.human_detector is None:
                    return self._create_error_result("人体检测器未初始化")
                human_results = self.human_detector.detect(frame)
                if not human_results:
                    return self._create_no_detection_result()
                human_bbox = human_results[0]["bbox"]

            # 双通道策略：尝试关键点ROI和BBox ROI
            roi_kp = None
            roi_bbox = None

            # 尝试关键点ROI
            if keypoints is not None and self.pose_detector is not None:
                roi_kp = self._optimize_head_roi_with_keypoints(
                    frame, human_bbox, keypoints
                )
                if roi_kp is not None:
                    self.logger.info("关键点ROI提取成功")

            # 备份BBox ROI
            if human_bbox is not None:
                roi_bbox = self._extract_head_roi_from_bbox(frame, human_bbox)
            else:
                roi_bbox = None
            if roi_bbox is not None:
                self.logger.info("BBox ROI提取成功")

            # 选择最佳ROI策略并更新统计
            head_roi = None
            head_roi_coords = None
            roi_strategy = None

            if roi_kp is not None and roi_bbox is not None:
                # 两个ROI都有效，优先使用关键点ROI（更精确）
                head_roi = roi_kp["roi"]
                head_roi_coords = roi_kp["coords"]
                roi_strategy = "keypoint"
                self.stats["keypoint_success"] += 1
                self.logger.info("使用关键点优化的头部ROI（双通道策略）")
            elif roi_kp is not None:
                # 只有关键点ROI有效
                head_roi = roi_kp["roi"]
                head_roi_coords = roi_kp["coords"]
                roi_strategy = "keypoint"
                self.stats["keypoint_success"] += 1
                self.logger.info("使用关键点优化的头部ROI")
            elif roi_bbox is not None:
                # 只有BBox ROI有效
                head_roi = roi_bbox["roi"]
                head_roi_coords = roi_bbox["coords"]
                roi_strategy = "bbox"
                self.stats["bbox_fallback"] += 1
                self.logger.info("使用BBox比例的头部ROI")
            else:
                # 两种方法都失败
                self.logger.warning("关键点和BBox ROI提取都失败")
                return self._create_no_detection_result()

            # 发网检测
            hairnet_result = self._simple_hairnet_detection(head_roi)

            # 计算关键点数量
            keypoint_count = 0
            if keypoints is not None:
                keypoint_count = len(keypoints)
                self.logger.info(f"检测到 {keypoint_count} 个关键点")

            # 添加头部ROI坐标信息和关键点数量
            hairnet_result["head_roi_coords"] = head_roi_coords
            hairnet_result["roi_strategy"] = roi_strategy
            hairnet_result["keypoint_count"] = keypoint_count

            # 添加调试信息
            self.logger.info(
                f"发网检测结果: 佩戴={hairnet_result.get('has_hairnet', False)}, "
                f"颜色={hairnet_result.get('hairnet_color', 'unknown')}, "
                f"置信度={hairnet_result.get('confidence', 0):.3f}, "
                f"像素数={hairnet_result.get('hairnet_pixels', 0)}, "
                f"像素密度={hairnet_result.get('pixel_density', 0):.3f}"
            )

            # 更新统计
            self.total_detections += 1
            if hairnet_result.get("has_hairnet", False):
                self.hairnet_detections += 1

            return hairnet_result

        except Exception as e:
            self.logger.error(f"发网检测失败: {e}")
            return self._create_error_result(str(e))

    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """
        创建错误结果

        Args:
            error_msg: 错误信息

        Returns:
            错误结果字典
        """
        return {
            "wearing_hairnet": False,
            "hairnet_color": "unknown",
            "confidence": 0.0,
            "hairnet_pixels": 0,
            "pixel_density": 0.0,
            "error": error_msg,
        }

    def _create_no_detection_result(self) -> Dict[str, Any]:
        """
        创建无检测结果

        Returns:
            无检测结果字典
        """
        return {
            "wearing_hairnet": False,
            "hairnet_color": "unknown",
            "confidence": 0.0,
            "hairnet_pixels": 0,
            "pixel_density": 0.0,
            "error": "no_detection",
        }

    def _analyze_hairnet_color(
        self, head_region: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        分析发网颜色

        Args:
            head_region: 头部区域图像
            mask: 发网区域掩码（可选）

        Returns:
            包含颜色分析结果的字典
        """
        try:
            if head_region.size == 0:
                return {"color": "unknown", "confidence": 0.0}

            # 转换为HSV颜色空间进行分析
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)

            # 如果有掩码，只分析掩码区域
            if mask is not None:
                hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask)
                # 计算掩码区域的主要颜色
                hist_h = cv2.calcHist([hsv_masked], [0], mask, [180], [0, 180])
            else:
                # 分析整个头部区域
                hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])

            # 找到主要色调
            dominant_hue = np.argmax(hist_h)

            # 根据色调判断颜色
            if dominant_hue < 15 or dominant_hue > 165:
                color = "white"
            elif 15 <= dominant_hue < 45:
                color = "blue"
            elif 45 <= dominant_hue < 75:
                color = "green"
            elif 75 <= dominant_hue < 105:
                color = "yellow"
            elif 105 <= dominant_hue < 135:
                color = "blue"
            else:
                color = "other"

            confidence = float(np.max(hist_h)) / np.sum(hist_h)

            return {
                "color": color,
                "confidence": confidence,
                "dominant_hue": int(dominant_hue),
            }

        except Exception as e:
            logger.error(f"颜色分析失败: {e}")
            return {"color": "unknown", "confidence": 0.0}

    def _detect_hairnet_with_pytorch(self, head_region: np.ndarray) -> Dict[str, Any]:
        """
        基于图像特征的简单发网检测（后备方案）

        Args:
            head_image: 头部区域图像

        Returns:
            检测结果字典
        """
        try:
            # 转换为灰度图
            if len(head_region.shape) == 3:
                gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
                # 保留原始彩色图像用于颜色分析
                color_image = head_region.copy()
            else:
                gray = head_region
                color_image = cv2.cvtColor(head_region, cv2.COLOR_GRAY2BGR)

            # 使用边缘检测寻找网状结构
            edges = cv2.Canny(gray, 50, 150)

            # 针对浅色发网的增强检测
            # 使用更敏感的边缘检测参数
            edges_sensitive = cv2.Canny(gray, 30, 120)

            # 颜色空间分析 - 检测浅色网状结构
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            # 检测浅蓝色范围 (扩大范围以提高检测率)
            light_blue_lower = np.array([90, 20, 80])  # 扩大浅蓝色下限
            light_blue_upper = np.array([140, 255, 255])  # 扩大浅蓝色上限
            light_blue_mask = cv2.inRange(hsv, light_blue_lower, light_blue_upper)

            # 检测其他浅色范围 (扩大范围)
            light_colors_lower = np.array([0, 0, 120])  # 降低浅色下限
            light_colors_upper = np.array([180, 120, 255])  # 扩大浅色上限
            light_mask = cv2.inRange(hsv, light_colors_lower, light_colors_upper)

            # 新增：检测白色和灰色发网
            white_lower = np.array([0, 0, 180])  # 白色下限
            white_upper = np.array([180, 30, 255])  # 白色上限
            white_mask = cv2.inRange(hsv, white_lower, white_upper)

            # 新增：检测绿色发网
            green_lower = np.array([40, 20, 80])  # 绿色下限
            green_upper = np.array([80, 255, 255])  # 绿色上限
            green_mask = cv2.inRange(hsv, green_lower, green_upper)

            # 结合颜色信息和边缘信息（包含所有颜色掩码）
            color_enhanced_edges = cv2.bitwise_or(edges, edges_sensitive)
            color_enhanced_edges = cv2.bitwise_or(color_enhanced_edges, light_blue_mask)
            color_enhanced_edges = cv2.bitwise_or(color_enhanced_edges, light_mask)
            color_enhanced_edges = cv2.bitwise_or(color_enhanced_edges, white_mask)
            color_enhanced_edges = cv2.bitwise_or(color_enhanced_edges, green_mask)

            # 计算基础边缘密度
            basic_edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            sensitive_edge_density = np.sum(edges_sensitive > 0) / (
                edges_sensitive.shape[0] * edges_sensitive.shape[1]
            )

            # 使用更好的边缘密度（取较大值）
            edge_density = max(basic_edge_density, sensitive_edge_density)

            # 计算颜色特征得分（包含所有颜色）
            light_blue_ratio = np.sum(light_blue_mask > 0) / (
                light_blue_mask.shape[0] * light_blue_mask.shape[1]
            )
            light_color_ratio = np.sum(light_mask > 0) / (
                light_mask.shape[0] * light_mask.shape[1]
            )
            white_ratio = np.sum(white_mask > 0) / (
                white_mask.shape[0] * white_mask.shape[1]
            )
            green_ratio = np.sum(green_mask > 0) / (
                green_mask.shape[0] * green_mask.shape[1]
            )

            # 综合颜色比例
            total_color_ratio = (
                light_blue_ratio + light_color_ratio + white_ratio + green_ratio
            )

            # 使用形态学操作检测网状结构（使用增强后的边缘）
            kernel = np.ones((3, 3), np.uint8)
            morphed = cv2.morphologyEx(color_enhanced_edges, cv2.MORPH_CLOSE, kernel)

            # 寻找轮廓
            contours, _ = cv2.findContours(
                morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 分析轮廓特征
            small_contours = [
                c
                for c in contours
                if cv2.contourArea(c) < 100 and cv2.contourArea(c) > 10
            ]

            # 发网检测逻辑（增强版）：
            # 1. 边缘密度适中（包含颜色增强）
            # 2. 有较多小轮廓（网状结构特征）
            # 3. 图像上部区域有更多边缘（发网通常在头顶）
            # 4. 浅色发网的颜色特征检测

            upper_half = color_enhanced_edges[: color_enhanced_edges.shape[0] // 2, :]
            upper_edge_density = np.sum(upper_half > 0) / (
                upper_half.shape[0] * upper_half.shape[1]
            )

            # 上部区域的颜色特征
            upper_light_blue = light_blue_mask[: light_blue_mask.shape[0] // 2, :]
            upper_light_blue_ratio = np.sum(upper_light_blue > 0) / (
                upper_light_blue.shape[0] * upper_light_blue.shape[1]
            )

            # 调整检测逻辑，增强对浅色发网的检测能力
            # 计算网状特征得分
            edge_score = min(1.0, edge_density * 6)  # 边缘密度得分（使用增强边缘）
            contour_score = min(1.0, len(small_contours) * 0.04)  # 轮廓数量得分
            upper_score = min(1.0, upper_edge_density * 5)  # 上部区域得分

            # 颜色特征得分（针对浅色发网）
            light_blue_score = min(1.0, light_blue_ratio * 20)  # 浅蓝色特征得分
            light_color_score = min(1.0, light_color_ratio * 15)  # 浅色特征得分
            upper_blue_score = min(1.0, upper_light_blue_ratio * 25)  # 上部浅蓝色得分

            # 综合得分（包含颜色特征）
            total_score = (
                edge_score
                + contour_score
                + upper_score
                + light_blue_score
                + light_color_score
                + upper_blue_score
            )

            # 智能检测阈值 - 平衡调整，减少误检但保持检测能力
            # 情况1：明显的浅蓝色发网特征（降低要求）
            has_light_blue_hairnet = (
                light_blue_ratio > 0.004
                and upper_light_blue_ratio > 0.003  # 降低浅蓝色区域要求
                and edge_density > 0.015  # 降低上部浅蓝色要求
                and len(small_contours) > 3  # 降低边缘特征要求
                and upper_edge_density > 0.018  # 降低轮廓要求  # 降低上部边缘密度要求
            )

            # 情况2：一般的发网检测（降低要求）
            has_general_hairnet = (
                total_score > 1.2
                and edge_density > 0.015  # 降低综合得分要求
                and len(small_contours) > 3  # 降低边缘密度要求
                and upper_edge_density > 0.018  # 降低轮廓数量要求
                and total_color_ratio > 0.006  # 降低上部区域边缘密度要求  # 降低颜色比例要求
            )

            # 情况3：浅色发网的特殊检测（降低要求）
            has_light_hairnet = (
                (
                    light_blue_ratio > 0.003
                    or light_color_ratio > 0.006
                    or white_ratio > 0.005
                    or green_ratio > 0.003
                )
                and edge_density > 0.012  # 降低颜色特征要求
                and len(small_contours) > 3  # 降低边缘要求
                and upper_edge_density > 0.015  # 降低轮廓要求
                and total_color_ratio > 0.004  # 降低上部边缘要求  # 降低总颜色比例要求
            )

            # 情况4：基础发网检测（降低要求）
            has_basic_hairnet = (
                edge_density > 0.012
                and len(small_contours) > 3  # 降低边缘要求
                and upper_edge_density > 0.015  # 降低轮廓要求
                and total_color_ratio > 0.004  # 降低上部边缘要求  # 降低综合颜色比例要求
            )

            # 情况5：最低检测标准（降低要求以提高检测率）
            has_minimal_hairnet = (
                (
                    edge_density > 0.010
                    and len(small_contours) > 2
                    and total_color_ratio > 0.004
                    and upper_edge_density > 0.012
                )
                or (  # 降低基本要求
                    total_color_ratio > 0.006
                    and upper_edge_density > 0.015
                    and len(small_contours) > 2
                    and edge_density > 0.010
                )
                or (  # 降低综合要求
                    light_color_ratio > 0.005
                    and edge_density > 0.010
                    and upper_edge_density > 0.012
                    and len(small_contours) > 2
                )
                or (  # 降低浅色要求
                    upper_edge_density > 0.018
                    and total_score > 1.0
                    and len(small_contours) > 3
                    and total_color_ratio > 0.004
                )  # 降低综合要求
            )

            # 最终判断：满足任一条件即可
            has_hairnet = (
                has_light_blue_hairnet
                or has_general_hairnet
                or has_light_hairnet
                or has_basic_hairnet
                or has_minimal_hairnet
            )

            # 计算置信度，包含颜色特征的综合评估
            if has_hairnet:
                # 基于多个特征的置信度计算（包含颜色特征）
                base_confidence = (
                    min(0.25, edge_density * 1.5)
                    + min(0.2, len(small_contours) * 0.01)  # 边缘密度贡献
                    + min(0.2, upper_edge_density * 1.2)  # 轮廓数量贡献  # 上部区域贡献
                )

                # 颜色特征置信度加成
                color_confidence = (
                    min(0.15, light_blue_ratio * 7.5)
                    + min(0.1, light_color_ratio * 3.3)  # 浅蓝色贡献
                    + min(0.1, upper_light_blue_ratio * 10)  # 浅色贡献  # 上部浅蓝色贡献
                )

                # 综合置信度
                confidence = base_confidence + color_confidence

                # 根据检测类型调整置信度
                if has_light_blue_hairnet:
                    confidence = min(0.9, confidence + 0.1)  # 浅蓝色发网加成
                elif has_light_hairnet:
                    confidence = min(0.85, confidence + 0.05)  # 浅色发网小幅加成

                confidence = min(0.95, confidence)  # 限制最大置信度
            else:
                confidence = 0.0

            return {
                "wearing_hairnet": bool(has_hairnet),  # 统一使用wearing_hairnet键
                "has_hairnet": bool(has_hairnet),  # 保持向后兼容
                "confidence": float(confidence),  # 确保是Python float类型
                "raw_prediction": int(1 if has_hairnet else 0),  # 确保是Python int类型
                "edge_density": float(edge_density),
                "contour_count": int(len(small_contours)),
                "hairnet_color": self._determine_hairnet_color(
                    float(light_blue_ratio),
                    float(light_color_ratio),
                    float(white_ratio),
                    float(green_ratio),
                ),
                "hairnet_pixels": int(
                    np.sum(light_blue_mask > 0)
                    + np.sum(light_mask > 0)
                    + np.sum(white_mask > 0)
                    + np.sum(green_mask > 0)
                ),
                "pixel_density": float(total_color_ratio),
                "debug_info": {
                    "basic_edge_density": float(basic_edge_density),
                    "sensitive_edge_density": float(sensitive_edge_density),
                    "light_blue_ratio": float(light_blue_ratio),
                    "light_color_ratio": float(light_color_ratio),
                    "white_ratio": float(white_ratio),
                    "green_ratio": float(green_ratio),
                    "total_color_ratio": float(total_color_ratio),
                    "upper_light_blue_ratio": float(upper_light_blue_ratio),
                    "upper_edge_density": float(upper_edge_density),
                    "total_score": float(total_score),
                    "has_light_blue_hairnet": bool(has_light_blue_hairnet),
                    "has_general_hairnet": bool(has_general_hairnet),
                    "has_light_hairnet": bool(has_light_hairnet),
                    "has_basic_hairnet": bool(has_basic_hairnet),
                    "has_minimal_hairnet": bool(has_minimal_hairnet),
                },
            }

        except Exception as e:
            logger.error(f"简单发网检测失败: {e}")
            return {
                "wearing_hairnet": False,
                "has_hairnet": False,
                "confidence": 0.0,
                "raw_prediction": 0,
                "edge_density": 0.0,
                "contour_count": 0,
                "hairnet_color": "unknown",
                "hairnet_pixels": 0,
                "pixel_density": 0.0,
                "debug_info": {},
            }

    def _determine_hairnet_color(
        self,
        light_blue_ratio: float,
        light_color_ratio: float,
        white_ratio: float,
        green_ratio: float,
    ) -> str:
        """根据颜色比例确定发网颜色"""
        color_ratios = {
            "light_blue": light_blue_ratio,
            "white": white_ratio,
            "green": green_ratio,
            "light": light_color_ratio,
        }

        # 找到最大比例的颜色
        max_color = max(color_ratios.keys(), key=lambda k: color_ratios[k])
        max_ratio = color_ratios[max_color]

        # 设置最小阈值
        if max_ratio > 0.0005:
            return max_color
        else:
            return "unknown"


class HairnetDetectionPipeline:
    """发网检测流水线

    整合人体检测和发网检测的完整流程
    """

    def __init__(self, person_detector, hairnet_detector: HairnetDetector):
        """
        初始化检测流水线

        Args:
            person_detector: 人体检测器实例
            hairnet_detector: 发网检测器实例
        """
        self.person_detector = person_detector
        self.hairnet_detector = hairnet_detector

        logger.info("HairnetDetectionPipeline initialized")

    def detect_hairnet_compliance(self, image):
        """
        检测图像中的发网佩戴合规性

        Args:
            image: 输入图像 (numpy array)

        Returns:
            dict: 包含检测结果的字典
        """
        try:
            # 检测人体
            persons = self.person_detector.detect(image)

            results = {
                "total_persons": len(persons),
                "persons_with_hairnet": 0,
                "persons_without_hairnet": 0,
                "compliance_rate": 0.0,
                "detections": [],
                "average_confidence": 0.0,
            }

            if len(persons) == 0:
                return results

            total_confidence = 0

            for person in persons:
                # 使用增强的头部ROI提取（如果可用）
                head_roi_result = None
                if (
                    hasattr(self.hairnet_detector, "use_enhanced_roi")
                    and self.hairnet_detector.use_enhanced_roi
                ):
                    # 尝试使用增强ROI提取
                    head_roi_result = self.hairnet_detector._extract_head_roi_enhanced(
                        image, person["bbox"], person.get("keypoints")
                    )

                # 如果增强方法失败，使用传统方法
                if head_roi_result is None:
                    x1, y1, x2, y2 = person["bbox"]
                    person_height = y2 - y1
                    head_height = int(person_height * 0.3)
                    head_region = image[y1 : y1 + head_height, x1:x2]
                    head_roi_result = (
                        {"roi": head_region, "coords": [x1, y1, x2, y1 + head_height]}
                        if head_region.size > 0
                        else None
                    )

                if head_roi_result is None:
                    logger.warning("头部区域提取失败，跳过此人员")
                    continue

                head_region = head_roi_result["roi"]
                head_coords = head_roi_result["coords"]

                # 发网检测 - 优先使用CNN模型
                try:
                    hairnet_result = self.hairnet_detector._detect_hairnet_with_cnn(
                        head_region
                    )
                except Exception as e:
                    logger.warning(f"CNN模型检测失败，回退到传统方法: {e}")
                    hairnet_result = self.hairnet_detector._detect_hairnet_with_pytorch(
                        head_region
                    )

                detection = {
                    "bbox": person["bbox"],
                    "head_coords": head_coords,
                    "roi_strategy": head_roi_result.get("strategy", "unknown"),
                    "has_hairnet": hairnet_result["has_hairnet"],
                    "confidence": hairnet_result["confidence"],
                    "hairnet_confidence": hairnet_result["confidence"],
                    "edge_density": hairnet_result.get("edge_density", 0.0),
                    "contour_count": hairnet_result.get("contour_count", 0),
                    "debug_info": hairnet_result.get("debug_info", {}),
                }

                results["detections"].append(detection)
                total_confidence += hairnet_result["confidence"]

                if hairnet_result["has_hairnet"]:
                    results["persons_with_hairnet"] += 1
                else:
                    results["persons_without_hairnet"] += 1

            # 计算合规率和平均置信度
            if results["total_persons"] > 0:
                results["compliance_rate"] = (
                    results["persons_with_hairnet"] / results["total_persons"]
                )
                results["average_confidence"] = (
                    total_confidence / results["total_persons"]
                )

            return results

        except Exception as e:
            logger.error(f"发网检测失败: {e}")
            return {
                "total_persons": 0,
                "persons_with_hairnet": 0,
                "persons_without_hairnet": 0,
                "compliance_rate": 0.0,
                "detections": [],
                "average_confidence": 0.0,
                "error": str(e),
            }

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        检测图像中的人员发网佩戴情况

        Args:
            frame: 输入图像

        Returns:
            检测结果列表
        """
        results = []

        # 1. 检测人体
        persons = self.person_detector.detect(frame)

        # 2. 对每个检测到的人进行发网检测
        for i, person in enumerate(persons):
            try:
                # 使用增强的头部ROI提取（如果可用）
                head_roi_result = None
                if (
                    hasattr(self.hairnet_detector, "use_enhanced_roi")
                    and self.hairnet_detector.use_enhanced_roi
                ):
                    # 尝试使用增强ROI提取
                    head_roi_result = self.hairnet_detector._extract_head_roi_enhanced(
                        frame, person["bbox"], person.get("keypoints")
                    )

                # 如果增强方法失败，使用传统方法
                if head_roi_result is None:
                    x1, y1, x2, y2 = person["bbox"]
                    person_height = y2 - y1
                    head_height = int(person_height * 0.3)
                    head_region = frame[y1 : y1 + head_height, x1:x2]
                    head_roi_result = (
                        {"roi": head_region, "coords": [x1, y1, x2, y1 + head_height]}
                        if head_region.size > 0
                        else None
                    )

                if head_roi_result is None:
                    logger.warning(f"人员 {i} 头部区域提取失败，跳过")
                    continue

                head_region = head_roi_result["roi"]
                head_coords = head_roi_result["coords"]

                # 发网检测 - 优先使用CNN模型
                try:
                    hairnet_result = self.hairnet_detector._detect_hairnet_with_cnn(
                        head_region
                    )
                except Exception as e:
                    logger.warning(f"CNN模型检测失败，回退到传统方法: {e}")
                    hairnet_result = self.hairnet_detector._detect_hairnet_with_pytorch(
                        head_region
                    )

                # 组合结果
                result = {
                    "person_id": i,
                    "bbox": person["bbox"],
                    "head_coords": head_coords,
                    "roi_strategy": head_roi_result.get("strategy", "unknown"),
                    "person_confidence": person["confidence"],
                    "has_hairnet": hairnet_result["has_hairnet"],
                    "hairnet_confidence": hairnet_result["confidence"],
                    "edge_density": hairnet_result.get("edge_density", 0.0),
                    "contour_count": hairnet_result.get("contour_count", 0),
                    "debug_info": hairnet_result.get("debug_info", {}),
                    "class_name": "person_with_hairnet"
                    if hairnet_result["has_hairnet"]
                    else "person_without_hairnet",
                }

                results.append(result)

            except Exception as e:
                logger.error(f"处理人员 {i} 时发生错误: {e}")
                continue

        logger.info(f"检测到 {len(results)} 个人员，发网佩戴检测完成")
        return results

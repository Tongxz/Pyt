import json
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class RegionType(Enum):
    """区域类型枚举"""

    ENTRANCE = "entrance"  # 入口区域
    HANDWASH = "handwash"  # 洗手区域
    SANITIZE = "sanitize"  # 消毒区域
    WORK_AREA = "work_area"  # 工作区域
    RESTRICTED = "restricted"  # 限制区域
    MONITORING = "monitoring"  # 监控区域


class Region:
    """区域类"""

    def __init__(
        self,
        region_id: str,
        region_type: RegionType,
        polygon: List[Tuple[int, int]],
        name: str = "",
    ):
        """
        初始化区域

        Args:
            region_id: 区域唯一标识
            region_type: 区域类型
            polygon: 区域多边形顶点列表 [(x1,y1), (x2,y2), ...]
            name: 区域名称
        """
        self.region_id = region_id
        self.region_type = region_type
        self.polygon = polygon
        self.name = name or f"{region_type.value}_{region_id}"
        self.is_active = True

        # 区域规则配置
        self.rules = {
            "required_behaviors": [],  # 必需的行为
            "forbidden_behaviors": [],  # 禁止的行为
            "max_occupancy": -1,  # 最大容纳人数 (-1表示无限制)
            "min_duration": 0.0,  # 最小停留时间
            "max_duration": -1.0,  # 最大停留时间 (-1表示无限制)
            "alert_on_violation": True,  # 违规时是否报警
        }

        # 统计信息
        self.stats = {
            "total_entries": 0,
            "current_occupancy": 0,
            "violations": 0,
            "last_entry_time": None,
            "last_exit_time": None,
        }

        logger.info(f"Region {self.name} created with {len(polygon)} vertices")

    def point_in_region(self, point: Tuple[int, int]) -> bool:
        """
        判断点是否在区域内（射线法）

        Args:
            point: 点坐标 (x, y)

        Returns:
            True if point is inside the region
        """
        x, y = point
        n = len(self.polygon)
        inside = False

        p1x, p1y = self.polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def bbox_in_region(self, bbox: List[int], threshold: float = 0.5) -> bool:
        """
        判断边界框是否在区域内

        Args:
            bbox: 边界框 [x1, y1, x2, y2]
            threshold: 重叠阈值（0.5表示边界框50%以上在区域内）

        Returns:
            True if bbox overlaps with region above threshold
        """
        x1, y1, x2, y2 = bbox

        # 检查边界框的关键点
        points = [
            (x1, y1),  # 左上
            (x2, y1),  # 右上
            (x1, y2),  # 左下
            (x2, y2),  # 右下
            ((x1 + x2) // 2, (y1 + y2) // 2),  # 中心
        ]

        inside_count = sum(1 for point in points if self.point_in_region(point))
        overlap_ratio = inside_count / len(points)

        return overlap_ratio >= threshold

    def set_rule(self, rule_name: str, value):
        """设置区域规则"""
        if rule_name in self.rules:
            self.rules[rule_name] = value
            logger.info(f"Region {self.name} rule '{rule_name}' set to {value}")
        else:
            logger.warning(f"Unknown rule '{rule_name}' for region {self.name}")

    def get_center(self) -> Tuple[float, float]:
        """获取区域中心点"""
        if not self.polygon:
            return (0, 0)

        x_coords = [point[0] for point in self.polygon]
        y_coords = [point[1] for point in self.polygon]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return (center_x, center_y)

    def get_area(self) -> float:
        """计算区域面积（鞋带公式）"""
        if len(self.polygon) < 3:
            return 0.0

        area = 0.0
        n = len(self.polygon)

        for i in range(n):
            j = (i + 1) % n
            area += self.polygon[i][0] * self.polygon[j][1]
            area -= self.polygon[j][0] * self.polygon[i][1]

        return abs(area) / 2.0

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "region_id": self.region_id,
            "region_type": self.region_type.value,
            "polygon": self.polygon,
            "name": self.name,
            "is_active": self.is_active,
            "rules": self.rules,
            "stats": self.stats,
        }


class RegionManager:
    """区域管理器

    管理所有检测区域，处理区域相关的逻辑
    """

    def __init__(self):
        """初始化区域管理器"""
        self.regions = {}  # region_id -> Region
        self.region_occupancy = {}  # region_id -> set of track_ids
        self.track_regions = {}  # track_id -> set of region_ids

        logger.info("RegionManager initialized")

    def add_region(self, region: Region) -> bool:
        """
        添加区域

        Args:
            region: 区域对象

        Returns:
            True if successfully added
        """
        if region.region_id in self.regions:
            logger.warning(f"Region {region.region_id} already exists")
            return False

        self.regions[region.region_id] = region
        self.region_occupancy[region.region_id] = set()

        logger.info(f"Region {region.name} added successfully")
        return True

    def remove_region(self, region_id: str) -> bool:
        """
        移除区域

        Args:
            region_id: 区域ID

        Returns:
            True if successfully removed
        """
        if region_id not in self.regions:
            logger.warning(f"Region {region_id} not found")
            return False

        # 清理相关数据
        del self.regions[region_id]
        if region_id in self.region_occupancy:
            del self.region_occupancy[region_id]

        # 清理追踪目标的区域记录
        for track_id in self.track_regions:
            self.track_regions[track_id].discard(region_id)

        logger.info(f"Region {region_id} removed successfully")
        return True

    def update_track_regions(self, track_id: int, bbox: List[int]) -> List[str]:
        """
        更新追踪目标所在的区域

        Args:
            track_id: 追踪目标ID
            bbox: 边界框

        Returns:
            当前所在的区域ID列表
        """
        current_regions = set()

        # 检查每个区域
        for region_id, region in self.regions.items():
            if not region.is_active:
                continue

            if region.bbox_in_region(bbox):
                current_regions.add(region_id)

        # 获取之前的区域
        previous_regions = self.track_regions.get(track_id, set())

        # 处理进入的区域
        entered_regions = current_regions - previous_regions
        for region_id in entered_regions:
            self._handle_region_entry(track_id, region_id)

        # 处理离开的区域
        exited_regions = previous_regions - current_regions
        for region_id in exited_regions:
            self._handle_region_exit(track_id, region_id)

        # 更新记录
        self.track_regions[track_id] = current_regions

        return list(current_regions)

    def _handle_region_entry(self, track_id: int, region_id: str):
        """处理区域进入事件"""
        region = self.regions[region_id]

        # 更新占用情况
        self.region_occupancy[region_id].add(track_id)

        # 更新统计信息
        region.stats["total_entries"] += 1
        region.stats["current_occupancy"] = len(self.region_occupancy[region_id])
        region.stats["last_entry_time"] = time.time()

        logger.info(f"Track {track_id} entered region {region.name}")

        # 检查容量限制
        max_occupancy = region.rules["max_occupancy"]
        if max_occupancy > 0 and region.stats["current_occupancy"] > max_occupancy:
            self._trigger_violation(region_id, track_id, "max_occupancy_exceeded")

    def _handle_region_exit(self, track_id: int, region_id: str):
        """处理区域离开事件"""
        region = self.regions[region_id]

        # 更新占用情况
        self.region_occupancy[region_id].discard(track_id)

        # 更新统计信息
        region.stats["current_occupancy"] = len(self.region_occupancy[region_id])
        region.stats["last_exit_time"] = time.time()

        logger.info(f"Track {track_id} exited region {region.name}")

    def _trigger_violation(self, region_id: str, track_id: int, violation_type: str):
        """触发违规事件"""
        region = self.regions[region_id]
        region.stats["violations"] += 1

        if region.rules["alert_on_violation"]:
            logger.warning(
                f"Violation in region {region.name}: {violation_type} by track {track_id}"
            )

    def check_behavior_compliance(self, track_id: int, behaviors: Dict) -> List[Dict]:
        """
        检查行为合规性

        Args:
            track_id: 追踪目标ID
            behaviors: 当前行为状态

        Returns:
            违规信息列表
        """
        violations = []

        if track_id not in self.track_regions:
            return violations

        for region_id in self.track_regions[track_id]:
            region = self.regions[region_id]

            # 检查必需行为
            for required_behavior in region.rules["required_behaviors"]:
                if (
                    required_behavior not in behaviors
                    or not behaviors[required_behavior].is_active
                ):
                    violation = {
                        "region_id": region_id,
                        "region_name": region.name,
                        "track_id": track_id,
                        "violation_type": "missing_required_behavior",
                        "details": f"Required behavior '{required_behavior}' not detected",
                    }
                    violations.append(violation)

            # 检查禁止行为
            for forbidden_behavior in region.rules["forbidden_behaviors"]:
                if (
                    forbidden_behavior in behaviors
                    and behaviors[forbidden_behavior].is_active
                ):
                    violation = {
                        "region_id": region_id,
                        "region_name": region.name,
                        "track_id": track_id,
                        "violation_type": "forbidden_behavior_detected",
                        "details": f"Forbidden behavior '{forbidden_behavior}' detected",
                    }
                    violations.append(violation)

        return violations

    def get_region_stats(self, region_id: str) -> Optional[Dict]:
        """获取区域统计信息"""
        if region_id not in self.regions:
            return None

        region = self.regions[region_id]
        return region.stats.copy()

    def get_all_regions_info(self) -> List[Dict]:
        """获取所有区域信息"""
        return [region.to_dict() for region in self.regions.values()]

    def save_regions_config(self, file_path: str) -> bool:
        """
        保存区域配置到文件

        Args:
            file_path: 配置文件路径

        Returns:
            True if successfully saved
        """
        try:
            config = {"regions": [region.to_dict() for region in self.regions.values()]}

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info(f"Regions config saved to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save regions config: {e}")
            return False

    def load_regions_config(self, file_path: str) -> bool:
        """
        从文件加载区域配置

        Args:
            file_path: 配置文件路径

        Returns:
            True if successfully loaded
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # 清空现有区域
            self.regions.clear()
            self.region_occupancy.clear()
            self.track_regions.clear()

            # 加载区域
            for region_data in config.get("regions", []):
                region_type = RegionType(region_data["type"])
                region = Region(
                    region_data["id"],
                    region_type,
                    region_data["points"],
                    region_data["name"],
                )
                region.is_active = region_data.get("is_active", True)
                region.rules = region_data.get("rules", region.rules)
                region.stats = region_data.get("stats", region.stats)

                self.add_region(region)

            logger.info(f"Regions config loaded from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load regions config: {e}")
            return False

    def reset(self):
        """重置区域管理器"""
        self.regions.clear()
        self.region_occupancy.clear()
        self.track_regions.clear()
        logger.info("RegionManager reset")

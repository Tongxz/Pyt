"""区域服务模块.

提供区域管理的服务层接口，包括区域的创建、更新、删除和查询功能.
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends

from core.region import Region, RegionManager, RegionType

logger = logging.getLogger(__name__)

# This would be initialized at app startup
region_manager: Optional[RegionManager] = None


def get_region_manager() -> Optional[RegionManager]:
    """获取区域管理器的FastAPI依赖项."""
    global region_manager
    if region_manager is None:
        logger.error("RegionManager is None in get_region_manager")
    return region_manager


class RegionService:
    """区域服务类.

    提供区域管理的业务逻辑接口.
    """

    def __init__(self, region_manager: RegionManager):
        """初始化区域服务.

        Args:
            region_manager: 区域管理器实例
        """
        self.region_manager = region_manager

    def get_all_regions(self) -> List[Dict[str, Any]]:
        """获取所有区域信息.

        Returns:
            区域信息列表
        """
        return self.region_manager.get_all_regions_info()

    def create_region(self, region_data: Dict[str, Any]) -> str:
        """创建新区域.

        Args:
            region_data: 区域数据字典

        Returns:
            创建的区域ID
        """
        region = self._from_dict(region_data)
        self.region_manager.add_region(region)
        return region.region_id

    def update_region(self, region_id: str, region_data: Dict[str, Any]):
        """更新指定区域.

        Args:
            region_id: 区域ID
            region_data: 更新的区域数据

        Raises:
            ValueError: 当区域不存在时
        """
        if region_id not in self.region_manager.regions:
            raise ValueError(f"Region with id {region_id} not found")

        updated_region = self._from_dict(region_data)
        updated_region.region_id = region_id

        self.region_manager.regions[region_id] = updated_region
        logger.info(f"Region {region_id} updated.")

    def delete_region(self, region_id: str):
        """删除指定区域.

        Args:
            region_id: 要删除的区域ID
        """
        self.region_manager.remove_region(region_id)

    def _from_dict(self, data: Dict[str, Any]) -> Region:
        """Helper to create a Region object from a dictionary."""
        try:
            region_type = RegionType(data["region_type"])
            region = Region(
                region_id=data["region_id"],
                region_type=region_type,
                polygon=data["polygon"],
                name=data.get("name", ""),
            )
            region.is_active = data.get("is_active", True)
            if "rules" in data:
                region.rules.update(data["rules"])
            return region
        except KeyError as e:
            raise ValueError(f"Missing key in region data: {e}")
        except Exception as e:
            raise ValueError(f"Error creating region from dict: {e}")


def get_region_service(
    region_manager: RegionManager = Depends(get_region_manager),
) -> "RegionService":
    """获取区域服务实例的FastAPI依赖项.

    Args:
        region_manager: 区域管理器依赖项

    Returns:
        区域服务实例

    Raises:
        RuntimeError: 当区域管理器未初始化时
    """
    if region_manager is None:
        raise RuntimeError("RegionManager not initialized")
    return RegionService(region_manager)


def initialize_region_service(regions_file: str):
    """初始化区域服务.

    Args:
        regions_file: 区域配置文件路径
    """
    global region_manager
    logger.info(f"Initializing RegionService with config: {regions_file}")
    try:
        region_manager = RegionManager()
        if not region_manager.load_regions_config(regions_file):
            logger.warning(
                f"Could not load regions from {regions_file}. Starting with an empty set."
            )
        logger.info("RegionService initialized successfully.")
        logger.info(f"Global region_manager set: {region_manager is not None}")
    except Exception as e:
        logger.exception(f"Failed to initialize RegionService: {e}")
        # 确保即使出错也有一个基本的RegionManager实例
        if region_manager is None:
            region_manager = RegionManager()
            logger.info("Created fallback RegionManager instance")

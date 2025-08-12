import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends

from core.region import Region, RegionManager, RegionType

logger = logging.getLogger(__name__)

# This would be initialized at app startup
region_manager: Optional[RegionManager] = None


def get_region_manager() -> Optional[RegionManager]:
    """FastAPI dependency to get the region manager."""
    return region_manager


class RegionService:
    def __init__(self, region_manager: RegionManager):
        self.region_manager = region_manager

    def get_all_regions(self) -> List[Dict[str, Any]]:
        return self.region_manager.get_all_regions_info()

    def create_region(self, region_data: Dict[str, Any]) -> str:
        region = self._from_dict(region_data)
        self.region_manager.add_region(region)
        return region.region_id

    def update_region(self, region_id: str, region_data: Dict[str, Any]):
        if region_id not in self.region_manager.regions:
            raise ValueError(f"Region with id {region_id} not found")

        updated_region = self._from_dict(region_data)
        updated_region.region_id = region_id

        self.region_manager.regions[region_id] = updated_region
        logger.info(f"Region {region_id} updated.")

    def delete_region(self, region_id: str):
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
    if region_manager is None:
        raise RuntimeError("RegionManager not initialized")
    return RegionService(region_manager)


def initialize_region_service(regions_file: str):
    global region_manager
    logger.info(f"Initializing RegionService with config: {regions_file}")
    try:
        region_manager = RegionManager()
        if not region_manager.load_regions_config(regions_file):
            logger.warning(
                f"Could not load regions from {regions_file}. Starting with an empty set."
            )
        logger.info("RegionService initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize RegionService: {e}")

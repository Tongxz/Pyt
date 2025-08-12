from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

# This would be in a service file
from services.region_service import RegionService, get_region_service

router = APIRouter()


@router.get("/regions", summary="获取所有区域信息")
def get_all_regions(
    region_service: RegionService = Depends(get_region_service),
) -> List[Dict[str, Any]]:
    return region_service.get_all_regions()


@router.post("/regions", summary="创建新区域")
def create_region(
    region_data: Dict[str, Any],
    region_service: RegionService = Depends(get_region_service),
) -> Dict[str, Any]:
    try:
        region_id = region_service.create_region(region_data)
        return {"status": "success", "region_id": region_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/regions/{region_id}", summary="更新区域信息")
def update_region(
    region_id: str,
    region_data: Dict[str, Any],
    region_service: RegionService = Depends(get_region_service),
) -> Dict[str, Any]:
    try:
        region_service.update_region(region_id, region_data)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/regions/{region_id}", summary="删除区域")
def delete_region(
    region_id: str, region_service: RegionService = Depends(get_region_service)
) -> Dict[str, Any]:
    try:
        region_service.delete_region(region_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

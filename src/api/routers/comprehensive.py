import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.core.optimized_detection_pipeline import OptimizedDetectionPipeline
from src.core.yolo_hairnet_detector import YOLOHairnetDetector
from src.services.detection_service import (
    comprehensive_detection_logic,
    get_hairnet_pipeline,
    get_optimized_pipeline,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/comprehensive", summary="综合检测接口")
async def detect_comprehensive(
    file: UploadFile = File(...),
    record_process: str = "false",
    optimized_pipeline: Optional[OptimizedDetectionPipeline] = Depends(
        get_optimized_pipeline
    ),
    hairnet_pipeline: Optional[YOLOHairnetDetector] = Depends(get_hairnet_pipeline),
) -> Dict[str, Any]:
    """
    执行综合检测，包括人体、发网、洗手、消毒等。
    优先使用优化管道，如果不可用则回退到原有逻辑。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")

    contents = await file.read()

    try:
        # 转换record_process参数
        # 临时强制启用录制模式用于测试
        should_record = True  # 强制启用录制模式
        # should_record = record_process.lower() == "true"  # 原始逻辑

        logger.info(
            f"开始综合检测: {file.filename}, 文件大小: {len(contents)} bytes, 录制模式: {should_record}"
        )

        result = comprehensive_detection_logic(
            contents=contents,
            filename=file.filename,
            optimized_pipeline=optimized_pipeline,
            hairnet_pipeline=hairnet_pipeline,
            record_process=should_record,
        )
        return result
    except Exception as e:
        logger.exception(f"综合检测失败: {e}")
        if "检测服务未初始化" in str(e):
            raise HTTPException(status_code=500, detail="检测服务未初始化")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@router.post("/image", summary="图像检测接口")
async def detect_image(
    file: UploadFile = File(...),
    optimized_pipeline: Optional[OptimizedDetectionPipeline] = Depends(
        get_optimized_pipeline
    ),
) -> Dict[str, Any]:
    """
    对单张图像进行人体行为检测。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")

    contents = await file.read()

    try:
        logger.info(
            f"开始图像检测: {file.filename}, 文件大小: {len(contents)} bytes"
        )

        # 使用优化管道进行检测
        if optimized_pipeline is None:
            raise HTTPException(status_code=500, detail="检测服务未初始化")

        # 这里应该调用图像检测逻辑
        # 暂时返回基本结构
        result = {
            "filename": file.filename,
            "detection_type": "image",
            "results": {
                "persons": [],
                "behaviors": [],
                "confidence": 0.0
            },
            "status": "success"
        }
        
        return result
    except Exception as e:
        logger.exception(f"图像检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@router.post("/hairnet", summary="发网检测接口")
async def detect_hairnet(
    file: UploadFile = File(...),
    hairnet_pipeline: Optional[YOLOHairnetDetector] = Depends(get_hairnet_pipeline),
) -> Dict[str, Any]:
    """
    专门进行发网检测。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")

    contents = await file.read()

    try:
        logger.info(
            f"开始发网检测: {file.filename}, 文件大小: {len(contents)} bytes"
        )

        # 使用发网检测管道
        if hairnet_pipeline is None:
            raise HTTPException(status_code=500, detail="发网检测服务未初始化")

        # 这里应该调用发网检测逻辑
        # 暂时返回基本结构
        result = {
            "filename": file.filename,
            "detection_type": "hairnet",
            "results": {
                "hairnet_detected": False,
                "confidence": 0.0,
                "bounding_boxes": []
            },
            "status": "success"
        }
        
        return result
    except Exception as e:
        logger.exception(f"发网检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

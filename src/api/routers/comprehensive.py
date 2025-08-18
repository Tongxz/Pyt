import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from core.optimized_detection_pipeline import OptimizedDetectionPipeline
from core.yolo_hairnet_detector import YOLOHairnetDetector
from services.detection_service import (
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

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/video/{filename}", summary="下载处理后的视频")
async def download_video(filename: str):
    """
    下载处理后的视频文件

    Args:
        filename: 视频文件名

    Returns:
        视频文件响应
    """
    # 视频文件存储目录
    video_dir = "./output/processed_videos"
    file_path = os.path.join(video_dir, filename)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        logger.warning(f"请求的视频文件不存在: {filename}")
        raise HTTPException(status_code=404, detail="视频文件不存在")

    # 检查文件是否在允许的目录内（安全检查）
    try:
        real_file_path = os.path.realpath(file_path)
        real_video_dir = os.path.realpath(video_dir)

        if not real_file_path.startswith(real_video_dir):
            logger.warning(f"尝试访问不安全的文件路径: {filename}")
            raise HTTPException(status_code=403, detail="访问被拒绝")
    except Exception as e:
        logger.error(f"文件路径安全检查失败: {e}")
        raise HTTPException(status_code=500, detail="文件访问错误")

    # 获取文件信息
    file_size = os.path.getsize(file_path)
    file_ext = Path(filename).suffix.lower()

    # 设置媒体类型
    media_type = "video/mp4"
    if file_ext == ".avi":
        media_type = "video/x-msvideo"
    elif file_ext == ".mov":
        media_type = "video/quicktime"
    elif file_ext == ".mkv":
        media_type = "video/x-matroska"

    logger.info(f"开始下载视频: {filename}, 大小: {file_size} bytes")

    # 返回文件响应
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
        headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"},
    )


@router.get("/image/{filename}", summary="下载处理后的图片")
async def download_image(filename: str):
    """
    下载处理后的图片文件

    Args:
        filename: 图片文件名

    Returns:
        图片文件响应
    """
    # 图片文件存储目录
    image_dir = "./output/processed_images"
    file_path = os.path.join(image_dir, filename)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        logger.warning(f"请求的图片文件不存在: {filename}")
        raise HTTPException(status_code=404, detail="图片文件不存在")

    # 检查文件是否在允许的目录内（安全检查）
    try:
        real_file_path = os.path.realpath(file_path)
        real_image_dir = os.path.realpath(image_dir)

        if not real_file_path.startswith(real_image_dir):
            logger.warning(f"尝试访问不安全的文件路径: {filename}")
            raise HTTPException(status_code=403, detail="访问被拒绝")
    except Exception as e:
        logger.error(f"文件路径安全检查失败: {e}")
        raise HTTPException(status_code=500, detail="文件访问错误")

    # 获取文件信息
    file_size = os.path.getsize(file_path)
    file_ext = Path(filename).suffix.lower()

    # 设置媒体类型
    media_type = "image/jpeg"
    if file_ext == ".png":
        media_type = "image/png"
    elif file_ext == ".gif":
        media_type = "image/gif"
    elif file_ext == ".bmp":
        media_type = "image/bmp"
    elif file_ext == ".webp":
        media_type = "image/webp"

    logger.info(f"开始下载图片: {filename}, 大小: {file_size} bytes")

    # 返回文件响应
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
        headers={"Content-Length": str(file_size)},
    )

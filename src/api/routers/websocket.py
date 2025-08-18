"""WebSocket路由模块.

提供实时图像检测的WebSocket端点.
"""
import asyncio
import base64
import json
import logging
from io import BytesIO

import cv2
import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from PIL import Image

from services.detection_service import comprehensive_detection_logic
from services.websocket_service import ConnectionManager, get_connection_manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, manager: ConnectionManager = Depends(get_connection_manager)
):
    """WebSocket检测端点."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            try:
                # 解析JSON消息
                message = json.loads(data)
                message_type = message.get("type")

                if message_type == "image":
                    # 处理图像检测请求
                    await process_image_detection(websocket, message)
                elif message_type == "pong":
                    # 心跳响应，不需要处理
                    pass
                else:
                    # 未知消息类型
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": f"Unknown message type: {message_type}",
                            }
                        )
                    )

            except json.JSONDecodeError:
                # JSON解析失败
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON format"})
                )
            except Exception as e:
                # 其他处理错误
                logger.error(f"WebSocket处理错误: {e}")
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "message": f"Processing error: {str(e)}"}
                    )
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from websocket.")


async def process_image_detection(websocket: WebSocket, message: dict):
    """处理图像检测请求."""
    try:
        # 获取base64图像数据
        image_data = message.get("data")
        if not image_data:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "No image data provided"})
            )
            return

        # 解码base64图像
        if image_data.startswith("data:image"):
            # 移除data URL前缀
            image_data = image_data.split(",")[1]

        # 解码base64
        image_bytes = base64.b64decode(image_data)

        # 转换为OpenCV图像
        image_pil = Image.open(BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # 调用检测服务
        detection_result = await asyncio.get_event_loop().run_in_executor(
            None, comprehensive_detection_logic, image_cv
        )

        # 提取检测结果
        total_persons = detection_result.get("total_persons", 0)
        statistics = detection_result.get("statistics", {})
        detections = detection_result.get("detections", [])

        # 构建WebSocket响应
        response = {
            "type": "comprehensive_detection_result",
            "detection_count": total_persons,
            "detections": detections,
            "statistics": {
                "persons_with_hairnet": statistics.get("persons_with_hairnet", 0),
                "persons_handwashing": statistics.get("persons_handwashing", 0),
                "persons_sanitizing": statistics.get("persons_sanitizing", 0),
            },
            "timestamp": asyncio.get_event_loop().time(),
        }

        # 发送检测结果
        await websocket.send_text(json.dumps(response))

    except Exception as e:
        logger.error(f"图像检测处理错误: {e}")
        await websocket.send_text(
            json.dumps(
                {"type": "error", "message": f"Image detection failed: {str(e)}"}
            )
        )

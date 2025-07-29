# FastAPI application
# FastAPI应用主文件

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
import asyncio
import json
import cv2
import numpy as np
from typing import Dict, Any, List
import base64
from io import BytesIO
from PIL import Image

# 导入检测器
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.detector import HumanDetector

# 创建FastAPI应用
app = FastAPI(
    title="人体行为检测系统 API",
    description="基于深度学习的实时人体行为检测与分析系统",
    version="1.0.0"
)

# 启用CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局检测器实例
detector = None

# WebSocket连接管理
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


# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化检测器"""
    global detector
    try:
        detector = HumanDetector()
        logger.info("检测器初始化成功")
    except Exception as e:
        logger.error(f"检测器初始化失败: {e}")
        raise

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "message": "服务运行正常",
        "detector_ready": detector is not None
    }

# API信息
@app.get("/api/v1/info")
async def get_api_info():
    """获取API信息"""
    return {
        "name": "人体行为检测系统 API",
        "version": "1.0.0",
        "description": "基于深度学习的实时人体行为检测与分析系统",
        "endpoints": {
            "health": "/health",
            "detect_image": "/api/v1/detect/image",
            "websocket": "/ws"
        }
    }

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
        annotated_image = detector.visualize_detections(image, results)
        
        # 将结果图像编码为base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 提取检测信息 - results已经是处理好的字典列表
        detections = results
        
        return {
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "annotated_image": f"data:image/jpeg;base64,{img_base64}"
        }
        
    except Exception as e:
        logger.error(f"图像检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")

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
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "消息格式错误"
                }))
                continue
            
            message_type = message.get("type")
            
            if message_type == "image":
                # 解码base64图像
                image_data = message.get("data", "")
                if not image_data:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "图像数据为空"
                    }))
                    continue
                
                if image_data.startswith("data:image"):
                    image_data = image_data.split(",")[1]
                
                try:
                    # 解码图像
                    img_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "图像解码失败"
                        }))
                        continue
                    
                    if detector is None:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "检测器未初始化"
                        }))
                        continue
                    
                    # 执行检测
                    results = detector.detect(image)
                    
                    # 发送检测结果
                    response = {
                        "type": "detection_result",
                        "detections": results,
                        "detection_count": len(results),
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await websocket.send_text(json.dumps(response))
                    
                except Exception as e:
                    logger.error(f"WebSocket检测失败: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"检测失败: {str(e)}"
                    }))
            
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

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
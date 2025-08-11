import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from services.websocket_service import ConnectionManager, get_connection_manager
from services.detection_service import comprehensive_detection_logic

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, manager: ConnectionManager = Depends(get_connection_manager)):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Here you can add logic to process real-time data
            # For example, decode image and run detection
            # For now, we just echo the data back
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from websocket.")
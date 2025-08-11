import asyncio
import json
import logging
from typing import List

from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.heartbeat_interval = 30  # Heartbeat interval (seconds)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connection established, current connections: {len(self.active_connections)}")
        # Start heartbeat task
        asyncio.create_task(self._heartbeat(websocket))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket connection disconnected, current connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if self.active_connections:
            for connection in self.active_connections.copy():
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.warning(f"Failed to broadcast message: {e}")
                    self.disconnect(connection)

    async def _heartbeat(self, websocket: WebSocket):
        """Heartbeat task to keep connection alive."""
        try:
            while websocket in self.active_connections:
                await asyncio.sleep(self.heartbeat_interval)
                if websocket in self.active_connections:
                    await websocket.send_text(json.dumps({"type": "ping"}))
        except Exception as e:
            logger.info(f"Heartbeat ended: {e}")
            self.disconnect(websocket)

# Singleton instance
manager = ConnectionManager()

def get_connection_manager() -> ConnectionManager:
    return manager
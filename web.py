from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from pydantic import BaseModel
import logging
from typing import Dict
from fastapi.websockets import WebSocketState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketRequest(BaseModel):
    message: str
    action: str = "message"  # default action is message
    session_id: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connection accepted for session: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"Removed session: {session_id}")
    
    async def send_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_text(message)
            logger.info(f"Sent message to session {session_id}: {message}")
    
    async def broadcast(self, message: str, exclude_session: str = None):
        for session_id, websocket in self.active_connections.items():
            if session_id != exclude_session:
                await websocket.send_text(message)
                logger.info(f"Broadcast message to session {session_id}: {message}")

router = APIRouter()
manager = ConnectionManager()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            request = WebSocketRequest(**data)
            logger.info(f"Received data from session {session_id}: {request}")
            
            # Handle close signal
            if request.action == "close":
                logger.info(f"Closing connection for session: {session_id}")
                break
                
            response = await some_function(request.message)
            await manager.send_message(response, session_id)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for session: {session_id}")
    finally:
        manager.disconnect(session_id)

async def some_function(data: str) -> str:
    # Replace this with your actual function logic
    import time
    time.sleep(2)
    return f"Processed: {data}"

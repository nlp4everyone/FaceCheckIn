# FastAPI Components
from fastapi import  APIRouter
from fastapi.responses import HTMLResponse
from fastapi import WebSocket
# Camera Feeder
from app.utils.camera import CameraFeeder
# Load config
from app.core.config import CAMERA_INDEX
# Other dependencies
import os, asyncio, time

# Check HTML file existed
camera_path = "app/templates/camera_feed.html"
if not os.path.exists(camera_path):
    raise FileNotFoundError(f"HMTL path: {camera_path} not found!")

# Define router
camera_route = APIRouter()
camera_feeder = CameraFeeder(camera_index = CAMERA_INDEX)

@camera_route.get("/")
async def get():
    with open("app/templates/camera_feed.html", "r") as f:
        html = f.read()
    return HTMLResponse(content=html)

@camera_route.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            begin = time.perf_counter()
            frame_bytes = camera_feeder.generate_frames(format=".jpg", quality = 80)
            if frame_bytes:
                await websocket.send_bytes(frame_bytes)
            await asyncio.sleep(0.01)  # ~30 FPS
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        camera_feeder.release()


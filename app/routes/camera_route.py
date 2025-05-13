# FastAPI Components
from fastapi import APIRouter, WebSocket, Request
from fastapi.templating import Jinja2Templates
# Camera Feeder
from app.utils.camera import CameraFeeder
from app.utils.image import ImageProcessing
# Load config
from app.core.config import CAMERA_INDEX
from app.core.constants import (CAMERA_QUALITY,
                                RECT_HEIGHT,
                                RECT_WIDTH)
# Other dependencies
import os, asyncio,json

# Check HTML file existed
camera_path = "app/templates/camera_feed.html"
if not os.path.exists(camera_path):
    raise FileNotFoundError(f"HMTL path: {camera_path} not found!")

# Define router
camera_route = APIRouter()
camera_feeder = CameraFeeder(camera_index = CAMERA_INDEX)
templates = Jinja2Templates(directory = "app/templates")

@camera_route.get("/")
async def get(request: Request):
    # Return
    return templates.TemplateResponse("camera_feed.html", {
        "request": request,
        "rect_width": RECT_WIDTH,
        "rect_height": RECT_HEIGHT
    })

@camera_route.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Read from camera
            frame_bytes, frame_numpy = camera_feeder.generate_frames(format=".jpg",
                                                                     quality = CAMERA_QUALITY)
            # Cropped centre frame
            cropped_frame = ImageProcessing.crop_centre_frame(frame = frame_numpy,
                                                              size = (RECT_WIDTH, RECT_HEIGHT))
            # Pseudo update status
            #await websocket.send_text(json.dumps({"status": "Please turn your head left"}))
            if frame_bytes:
                await websocket.send_bytes(frame_bytes)
            await asyncio.sleep(0.01)  # ~30 FPS
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        camera_feeder.release()


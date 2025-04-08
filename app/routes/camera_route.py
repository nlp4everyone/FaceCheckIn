# FastAPI Components
from fastapi import  APIRouter
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
# Camera Feeder
from app.utils.camera import CameraFeeder
# Load config
from app.core.config import CAMERA_INDEX
# Other dependencies
import os

# Check HTML file existed
camera_path = "app/templates/camera_feed.html"
if not os.path.exists(camera_path):
    raise FileNotFoundError(f"HMTL path: {camera_path} not found!")

# Define router
camera_route = APIRouter()
camera_feeder = CameraFeeder(camera_index = CAMERA_INDEX)

# Set up Jinja2 templates directory
templates = Jinja2Templates(directory="app/templates")

@camera_route.get("/")
async def index(request: Request):
    return templates.TemplateResponse("camera_feed.html", {"request": request})


@camera_route.get("/video_feed")
async def video_feed():
    return StreamingResponse(camera_feeder.generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

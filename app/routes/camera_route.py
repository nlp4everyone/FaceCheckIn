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
                                RECT_WIDTH,
                                FACE_WAIT_TIME)
# Getting model
from app.startup import get_face_recognition_model
# Other dependencies
import os, asyncio, json

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
    mtcnn = get_face_recognition_model()
    collecting = False
    first_detected_time = None
    collected_detections = []

    try:
        while True:
            # Read from camera
            frame_bytes, frame_numpy = camera_feeder.generate_frames(format=".jpg",
                                                                     quality = CAMERA_QUALITY)
            # Cropped centre frame
            cropped_frame = ImageProcessing.crop_centre_frame(frame = frame_numpy,
                                                              size = (RECT_WIDTH, RECT_HEIGHT))
            try:
                # Handle face detection
                detections = mtcnn.detect_faces(cropped_frame)
                # When detection existed
                if detections:
                    if not collecting:
                        # First detection arrives
                        collecting = True
                        # Declare first appearance time
                        first_detected_time = asyncio.get_event_loop().time()
                        # Send notification
                        await websocket.send_text(json.dumps({"status": f"Stop your motion for {int(FACE_WAIT_TIME)} second"}))
                        collected_detections = [detections]
                    else:
                        collected_detections.append(detections)
            except:
                pass  # No face detected this frame

            # Check if 1 second has passed since first detection
            if collecting and (asyncio.get_event_loop().time() - first_detected_time) >= FACE_WAIT_TIME:
                # Process collected detections
                print(f"Collected {len(collected_detections)} detections in {int(FACE_WAIT_TIME)} second")
                print(collected_detections[0])
                # 👉 Process collected_detections here
                # Reset state
                collecting = False
                collected_detections.clear()
                # Send notification back
                await websocket.send_text(json.dumps({"status": "Please place your face inside a red area"}))

            if frame_bytes:
                await websocket.send_bytes(frame_bytes)
            await asyncio.sleep(0.01)  # ~30 FPS
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        camera_feeder.release()


# FastAPI Components
from fastapi import APIRouter, WebSocket, Request
from fastapi.templating import Jinja2Templates
# Camera Feeder
from app.utils.camera import CameraFeeder
from app.utils import is_selected_image
# Load config
from app.core.config import CAMERA_INDEX
from app.core.constants import (CAMERA_QUALITY,
                                RECT_HEIGHT,
                                RECT_WIDTH,
                                FACE_WAIT_TIME,
                                FRAME_SKIPPING_ITERATION,
                                MIN_ACCEPTED_FPS)
from app.utils.face.frontal_metrics import FrontalFaceFiltering
# Getting model
from app.startup import get_face_recognition_model
# Other dependencies
import os, asyncio, json, cv2, uuid

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

    # Init models
    mediapipe = get_face_recognition_model()
    # Init params
    collecting = False
    first_detected_time = None
    total_frames = 0
    # Init results
    face_frames = []
    collected_detections = []

    try:
        while True:
            # Read from camera
            frame_bytes, frame_numpy = camera_feeder.generate_frames(format=".jpg",
                                                                     quality = CAMERA_QUALITY)
            # Accumulate frames
            total_frames +=1

            try:
                # Add Fixed Frame Skipping for better speed and offload CPU
                if total_frames % FRAME_SKIPPING_ITERATION == 0:
                    # Handle face detection
                    detections = mediapipe.detect_faces(frame_numpy)
                    # When detection existed and archive desired IOU
                    if detections and is_selected_image(frame_numpy,tuple(detections[0].box),RECT_WIDTH,RECT_HEIGHT):
                        if not collecting:
                            # First detection arrives
                            collecting = True
                            # Declare first appearance time
                            first_detected_time = asyncio.get_event_loop().time()
                            # Send notification
                            await websocket.send_text(json.dumps({"status": f"Stop your motion for {int(FACE_WAIT_TIME)} second"}))
                            face_frames = [frame_numpy]
                            collected_detections = detections
                        else:
                            face_frames.append(frame_numpy)
                            collected_detections.extend(detections)

            except:
                pass  # No face detected this frame

            # Send frame to websocket
            if frame_bytes:
                await websocket.send_bytes(frame_bytes)
            await asyncio.sleep(0.01)  # ~30 FPS

            total_detected_frames = len(face_frames)
            # Check if FACE_WAIT_TIME second has passed since first detection
            if collecting and (asyncio.get_event_loop().time() - first_detected_time) >= FACE_WAIT_TIME:
                # Reset state
                collecting = False
                # For assuring enough frame for processing by archiving minimum FPS
                current_fps = total_detected_frames / FACE_WAIT_TIME
                # If current FPS less than MIN ACCEPTED FPS, skip turn
                if current_fps < MIN_ACCEPTED_FPS:
                    # Send notification
                    await websocket.send_text(json.dumps({"status": "Too quick, please slow down your motion!"}))
                    # Reset total frames
                    total_frames = 0
                    continue

                # Process collected detections
                print(f"Collected {total_detected_frames} detections in {int(FACE_WAIT_TIME)} second")
                # Select most frontal face image from input
                selected_frames = FrontalFaceFiltering.select_frames(frames = face_frames,
                                                                     detections = collected_detections)
                # Saved frame to Disk
                # cv2.imwrite(f"{uuid.uuid4()}.png", selected_frames[0])
                # Reset state
                face_frames.clear()
                # Send notification back
                await websocket.send_text(
                    json.dumps({"status": "Please place your face inside a red area"}))

    except Exception as e:
        print("WebSocket error:", e)
    finally:
        camera_feeder.release()


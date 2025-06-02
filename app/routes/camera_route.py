# FastAPI Components
import numpy as np
from fastapi import APIRouter, WebSocket, Request
from fastapi.templating import Jinja2Templates
# Camera Feeder
from app.utils.camera import CameraFeeder
from app.utils import is_standard_image
# Load config
from app.core.config import CAMERA_INDEX
from app.core.config.constants import *
# Load message content
from app.core.status_message import *
# Detection filter
from app.utils.face.frontal_metrics import FrontalFaceFiltering
from app.utils.face.alignment import BasicAlignment
from app.utils.face.embedding import calculate_similarity
# Getting model
from app.startup import (get_face_recognition_model,
                         get_face_embedding_model,
                         get_qdrant_service,
                         get_minio_storage)
# Timer
from app.utils.timer import Timer
# Image component
from app.utils.image import ImagePreprocess
# Logger
from loggers import SystemLogger
# Other dependencies
import os, asyncio, json
# Date time
from datetime import datetime
from typing import List

# Check HTML file existed
camera_path = "app/templates/camera_feed.html"
if not os.path.exists(camera_path):
    raise FileNotFoundError(f"HMTL path: {camera_path} not found!")

# Define router
camera_route = APIRouter()
# Camera Feeder
camera_feeder = CameraFeeder(camera_index = CAMERA_INDEX)
# Template
templates = Jinja2Templates(directory = "app/templates")
# Timer
timer = Timer("Data Pipeline")

@camera_route.get("/", include_in_schema = False)
async def get(request: Request):
    # Return
    return templates.TemplateResponse("camera_feed.html", {
        "request": request,
        "rect_width": RECT_WIDTH,
        "rect_height": RECT_HEIGHT
    })

async def send_delayed_notification(msg :dict,
                                    websocket :WebSocket,
                                    delay_time :int = 5):
    # Schedule follow-up message 5 seconds later (non-blocking)
    await asyncio.sleep(delay_time)
    await websocket.send_text(json.dumps(msg))

@camera_route.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Init models
    mediapipe = get_face_recognition_model()
    embedding_model = get_face_embedding_model()
    qdrant_service = get_qdrant_service()
    minio_storage = get_minio_storage()

    # Init params
    collecting = False
    first_detected_time = None
    total_frames = 0
    # Init results
    face_frames = []
    collected_detections = []
    frames :List[np.ndarray] = []
    last_face_embedding = None
    last_alert_time = None
    last_detected_time = None

    try:
        while True:
            # Read from camera
            frame_bytes, frame_numpy = camera_feeder.generate_frames(format = ".jpg",
                                                                     quality = CAMERA_QUALITY)

            # Resized to fixed size image (Reducing time for processing)
            frame_numpy, resized_ratio = ImagePreprocess.resize_image_keep_aspect_ratio(frame_numpy,
                                                                                        fixed_width = DOWNSCALE_IMAGE_WIDTH)
            # Accumulate frames
            total_frames +=1
            if collecting: frames.append(frame_numpy)

            # Check if FACE_WAIT_TIME second has passed since first detection
            current_time = asyncio.get_event_loop().time()

            # Skip if in cooldown
            in_spam_cooldown = last_alert_time is not None and (current_time - last_alert_time) < SPAM_COOLDOWN_SECONDS
            # Send frame to websocket
            if frame_bytes:
                await websocket.send_bytes(frame_bytes)
            await asyncio.sleep(0.01)  # ~30 FPS

            try:
                # Add Fixed Frame Skipping for better speed and offload CPU
                if total_frames % FRAME_SKIPPING_ITERATION == 0:
                    # Handle face detection
                    detections = mediapipe.detect_faces(frame_numpy)


                    # Specify whether image choosen or not
                    standarded_image = is_standard_image(frame_numpy,
                                                         face_bbox = tuple(detections[0].box),
                                                         rect_width = int(RECT_WIDTH * resized_ratio),
                                                         rect_height = int(RECT_HEIGHT * resized_ratio),
                                                         accepted_threshold = IOU_ACCEPTED_THRESHOLD)
                    # When detection existed and archive desired IOU
                    if detections and standarded_image:
                        # By pass
                        if not collecting:
                            # Set minimum time between success check in time
                            if last_detected_time and current_time - last_detected_time < MINIMUM_INSPECT_DURATION:
                                collecting = False
                                continue

                            collecting = True
                            # Declare first appearance time
                            first_detected_time = asyncio.get_event_loop().time()
                            if not in_spam_cooldown:
                                # Send notification
                                await websocket.send_text(json.dumps(DETECTION_START_MSG(FACE_WAIT_TIME)))
                            face_frames = [frame_numpy]
                            collected_detections = detections
                        else:
                            face_frames.append(frame_numpy)
                            collected_detections.extend(detections)

            except:
                pass  # No face detected this frame

            total_detected_frames = len(face_frames)

            # Wait util reach wait time
            if collecting and (current_time - first_detected_time) >= FACE_WAIT_TIME:
                # Reset state
                collecting = False
                # For assuring enough frame for processing by archiving minimum FPS
                current_fps = total_detected_frames / FACE_WAIT_TIME

                # When in Cool Down duration, do nothing util it expires.
                if in_spam_cooldown:
                    # Send spamming alert to screen
                    await websocket.send_text(json.dumps(SPAMMING_MSG))
                    # Reset state
                    face_frames.clear()
                    frames.clear()
                    continue

                # If current FPS less than MIN ACCEPTED FPS, skip turn
                if current_fps < MIN_ACCEPTED_FPS:
                    # Send notification
                    await websocket.send_text(json.dumps(QUICK_MOTION_MSG))
                    # Reset state
                    face_frames.clear()
                    frames.clear()
                    # Reset total frames
                    total_frames = 0
                    continue

                # Start timer
                timer.start()
                # Process collected detections
                SystemLogger.success(f"Collected {total_detected_frames} detections in {int(FACE_WAIT_TIME)} second")
                # Select most frontal face image from input
                selected_frames, detection_results = FrontalFaceFiltering.select_frames(frames = face_frames,
                                                                                        detections = collected_detections)
                # Mark time
                timer.mark("Sorting frame duration")

                # Keypoint
                keypoint = detection_results[0][1].model_dump().get("keypoint")
                # Aligned image
                aligned_image = BasicAlignment.align_face_5points(image = selected_frames[0],
                                                                  landmarks = keypoint)
                # Mark
                timer.mark("Alignment duration")

                # Embedding
                face_embedding = embedding_model.embed(aligned_image)
                # Mark
                timer.mark("Embedding duration")

                # When last face is not empty, compare
                if last_face_embedding is not None:
                    # Get similiar score
                    similarity_score = calculate_similarity(face_embedding, last_face_embedding).float()
                    # Prevent repeated spamming
                    if last_face_embedding is not None and similarity_score >= FACE_SIMILARITY_THRESHOLD:
                        # Define last alert time
                        last_alert_time = asyncio.get_event_loop().time()
                # Retrieve face
                face_retrieved = await qdrant_service.retrieve_points(face_embedding[0].tolist(),
                                                                      score_threshold = FACE_SIMILARITY_THRESHOLD)
                # Mark
                timer.mark("Retrieving duration")

                # Make report
                timer.stop()
                reports = timer.report()
                # Logger
                for report in reports: SystemLogger.info(report)

                identfication = "Undefined"
                # When found face
                if face_retrieved:
                    user_info = face_retrieved[0].payload
                    # Get face name
                    identfication = user_info.get("face_name")
                    # Send text back to screen
                    await websocket.send_text(json.dumps(FACE_SINGED_MSG(identfication)))
                else:
                    # Opposite
                    await websocket.send_text(json.dumps(FACE_NOT_FOUND_MSG))

                # Define timestamp
                file_name = datetime.now().strftime("%Y/%m/%d/%Y-%m-%d_%H:%M:%S")

                # Saved data in background
                await minio_storage.aupload_image(bucket_name = MINIO_CHECKIN_BUCKET,
                                                  image = aligned_image,
                                                  image_name = f"{file_name}_{identfication}.png")

                # Saved data in background
                await minio_storage.aupload_video(bucket_name = MINIO_CHECKIN_BUCKET,
                                                  images = frames,
                                                  video_name = f"{file_name}_{identfication}.mp4")
                # Last detected time
                last_detected_time = asyncio.get_event_loop().time()
                # Set value to last embedding
                last_face_embedding = face_embedding

                # Reset state
                face_frames.clear()
                frames.clear()

    except Exception as e:
        SystemLogger.error(e)
    finally:
        camera_feeder.release()


from fastapi import FastAPI
# Define route
from .routes import (camera_route,
                     preview_router,
                     face_management_router)
# Define startup
from .startup import (init_models,
                      init_qdrant_service,
                      init_minio_storage)
# Components
from loggers import SystemLogger
import time

# Tags
tags_metadata = [
    {
        "name": "Development",
        "description": "Contain features such as face alignment, face comparison for development step",
    },
    {
        "name": "Face Management",
        "description": "Contain features such as face register, face delete, face retrieve",
    },
]

app = FastAPI()
# Append route
app.include_router(camera_route)
app.include_router(preview_router,
                   prefix = "/development",
                   tags = [tags_metadata[0].get("name")])
app.include_router(face_management_router,
                   tags = [tags_metadata[1].get("name")])


@app.on_event("startup")
async def startup_event():
    # Start
    start = time.perf_counter()
    # Init ml model
    init_models()
    # Init Qdrant
    qdrant_service = init_qdrant_service()
    await qdrant_service.create_collection()
    # Init minio
    init_minio_storage()
    # Measure time for processing
    SystemLogger.success(f"Start up done after: {round(time.perf_counter() - start, 1)}s")
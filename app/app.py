from fastapi import FastAPI
# Define route
from .routes import camera_route, preview_router
# Define startup
from .startup import (init_models)
# Components
from loggers import SystemLogger
import time

app = FastAPI()
# Append route
app.include_router(camera_route)
app.include_router(preview_router)

@app.on_event("startup")
async def startup_event():
    # Start
    start = time.perf_counter()
    # Init ml model
    init_models()
    # Measure time for processing
    SystemLogger.success(f"Start up done after: {round(time.perf_counter() - start, 1)}s")
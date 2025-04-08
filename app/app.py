from fastapi import FastAPI
# Define route
from .routes import camera_route

app = FastAPI()
# Append route
app.include_router(camera_route)
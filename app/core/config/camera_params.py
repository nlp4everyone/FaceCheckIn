from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()
# Load params
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX"))
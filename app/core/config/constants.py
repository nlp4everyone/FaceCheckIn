# Image params
CAMERA_QUALITY :int = 70
RECT_WIDTH :int = 200
RECT_HEIGHT :int = 250

# System params
FACE_WAIT_TIME :float = 3.0
SPAM_COOLDOWN_SECONDS :int = 10
MIN_ACCEPTED_FPS :int = 5 # Minimum FPS each turn for avoiding quick motion
FACE_SIMILARITY_THRESHOLD :float = 0.35
IOU_ACCEPTED_THRESHOLD :float = 0.25
MINIMUM_INSPECT_DURATION :int = 3

# Optimization params
FRAME_SKIPPING_ITERATION :int = 2 # Default is 1, which use all frames for detecting. Increase for skipp
DOWNSCALE_IMAGE_WIDTH :int = 512 # Image width shape will apply to downscale image
COMPRESS_IMAGE_RATIO :int = 70 # Number in percent of quality remaining for image

# Qdrant Default config
FACE_EMBEDDING_DIMS :int = 512
DEFAULT_SIMILARITY_TOP_K :int = 3

# Minio default config
MINIO_REGISTERED_BUCKET :str = "registered-face"
MINIO_CHECKIN_BUCKET :str = "checkin-face"




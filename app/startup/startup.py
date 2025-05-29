# init model
from app.utils.face.embedding import TimmEmbedding, BaseEmbedding
from app.utils.face.recognition import MediapipeDetection
# Db service
from app.db.qdrant import QdrantService, Distance
from app.db.minio import MinioObjectStorage

# Config
from app.core.config.constants import (FACE_EMBEDDING_DIMS,
                                       MINIO_REGISTERED_BUCKET)
from app.core.config import (MINIO_ACCESS_KEY,
                             MINIO_SECRET_KEY,
                             MINIO_HOST,
                             QDRANT_HOST)

# # Variable
# mtcnn = None
# ada_face = None
# # Variable
# face_embedding_model = None
# mediapipe = None
# qdrant_service = None
# registered_minio = None

def init_models():
    """Start Postgres Connection"""
    # global mtcnn
    global mediapipe
    global frontal_face_filtering
    global face_embedding_model
    # # Init connection
    # mtcnn = MTCNNRecognition(device = "cpu",
    #                          post_process = False)
    mediapipe = MediapipeDetection(model_selection = 0,
                                   min_detection_confidence = 0.8)
    # face_embedding_model = AdaFaceEmbedding(model_name = EMBEDDING_MODEL,
    #                                         HF_TOKEN = HF_KEY)
    face_embedding_model = TimmEmbedding(device = "cpu")
    return mediapipe

def init_qdrant_service() -> QdrantService:
    global qdrant_service
    qdrant_service = QdrantService(host = QDRANT_HOST,
                                   embedding_dims = FACE_EMBEDDING_DIMS,
                                   distance = Distance.COSINE)
    return qdrant_service

def init_minio_storage():
    global registered_minio
    registered_minio = MinioObjectStorage(endpoint = f"{MINIO_HOST}:9000",
                                          bucket_name = MINIO_REGISTERED_BUCKET,
                                          access_key = MINIO_ACCESS_KEY,
                                          secret_key = MINIO_SECRET_KEY)
def get_face_embedding_model() ->BaseEmbedding:
    return face_embedding_model

def get_face_recognition_model() ->MediapipeDetection:
    return mediapipe

def get_qdrant_service() -> QdrantService:
    return qdrant_service

def get_registered_minio() -> MinioObjectStorage:
    return registered_minio



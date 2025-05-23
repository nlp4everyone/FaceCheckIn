# init model
# from app.utils.face.embedding import AdaFaceEmbedding, TimmEmbedding
from app.utils.face.recognition import MTCNNRecognition, MediapipeDetection

# Variable
mtcnn = None
ada_face = None
# Variable
face_embedding_model = None
mediapipe = None

def init_models():
    """Start Postgres Connection"""
    # global mtcnn
    global mediapipe
    global frontal_face_filtering
    # global face_embedding_model
    # # Init connection
    # mtcnn = MTCNNRecognition(device = "cpu",
    #                          post_process = False)
    mediapipe = MediapipeDetection(model_selection = 0,
                                   min_detection_confidence = 0.8)
    # face_embedding_model = AdaFaceEmbedding(model_name = EMBEDDING_MODEL,
    #                                         HF_TOKEN = HF_TOKEN)
    # face_embedding_model = TimmEmbedding(device = "cuda")
    return mediapipe


# def get_face_embedding_model():
#     return face_embedding_model

def get_face_recognition_model() ->MediapipeDetection:
    return mediapipe



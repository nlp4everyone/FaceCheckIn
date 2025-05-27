# init model
from app.utils.face.embedding import TimmEmbedding, BaseEmbedding
from app.utils.face.recognition import MediapipeDetection

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
    global face_embedding_model
    # # Init connection
    # mtcnn = MTCNNRecognition(device = "cpu",
    #                          post_process = False)
    mediapipe = MediapipeDetection(model_selection = 0,
                                   min_detection_confidence = 0.8)
    # face_embedding_model = AdaFaceEmbedding(model_name = EMBEDDING_MODEL,
    #                                         HF_TOKEN = HF_KEY)
    face_embedding_model = TimmEmbedding(device = "cpu",
                                         model_name="hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3")
    return mediapipe


def get_face_embedding_model() ->BaseEmbedding:
    return face_embedding_model

def get_face_recognition_model() ->MediapipeDetection:
    return mediapipe



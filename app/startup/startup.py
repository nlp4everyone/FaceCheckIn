# init model
# from app.utils.face.embedding import AdaFaceEmbedding, TimmEmbedding
from app.utils.face.recognition import MTCNNRecognition

# Variable
mtcnn = None
ada_face = None
# Variable
face_embedding_model = None

def init_models():
    """Start Postgres Connection"""
    global mtcnn
    global face_embedding_model
    # Init connection
    mtcnn = MTCNNRecognition(device = "cpu",
                             post_process = False)
    # face_embedding_model = AdaFaceEmbedding(model_name = EMBEDDING_MODEL,
    #                                         HF_TOKEN = HF_TOKEN)
    # face_embedding_model = TimmEmbedding(device = "cuda")
    return mtcnn


def get_face_embedding_model():
    return face_embedding_model

def get_face_recognition_model() ->MTCNNRecognition:
    return mtcnn


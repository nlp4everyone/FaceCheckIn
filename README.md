# 🛸 Introduction:

A web-UI application based on FastAPI, supports in face detection and face verification. 

<br />


# 🔗 Installing:
1. Clone this project:
```
git clone -b baseline https://github.com/nlp4everyone/face_ekyc
```
2. Go inside project:
```
cd face_ekyc
```
3. Create .env file from .env.sample, then change env variables if necessary:
```
cp .env.sample .env
```
4. Connect your camera to PC, then change CAMERA_INDEX correspondingly in .env file.
5. Install requirements (Install alongside Anaconda if necessary):
```
pip install -r requirements.txt
```
7. Open up the application:
```
uvicorn app.app:app
```

# 📃 Intergrations:
- 🖥️ Web Framework: FastAPI
- 🗃️ Vector Store: Qdrant
- 📂 Image Store: Minio
- 🎮 Model: Face Detection 🤫 (Mediapipe) for CPU usage, Face Embedding 🤗 (ArcFace, AdaFace)
<br />

# 📔 Related Reference:
- 📖 ArcFace: https://arxiv.org/pdf/1801.07698
- 📖 AdaFace: https://arxiv.org/pdf/2204.00964
<br />

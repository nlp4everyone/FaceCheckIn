import cv2

class CameraFeeder:
    def __init__(self,
                 camera_index :int = 0):
        # Define property
        self._camera = cv2.VideoCapture(camera_index)
        # Check status
        if not self._camera.isOpened():
            raise IndexError(f"Failed to open camera at index {camera_index}")

    def generate_frames(self):
        while True:
            success, frame = self._camera.read()
            if not success:
                break

            # Encode
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            # Frame bytes
            frame_bytes = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )
            # Return
            yield frame_bytes
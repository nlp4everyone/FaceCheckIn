import cv2, base64

class CameraFeeder:
    def __init__(self,
                 camera_index :int = 0):
        # Define property
        self._camera = cv2.VideoCapture(camera_index)
        # Check status
        if not self._camera.isOpened():
            raise IndexError(f"Failed to open camera at index {camera_index}")

    def generate_frames(self,
                        format: str = '.webp',
                        quality: int = 80):
        while True:
            success, frame = self._camera.read()
            if not success:
                break

            # Choose encoder settings
            if format == '.jpg':
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            elif format == '.webp':
                encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
            else:
                raise ValueError("Unsupported format. Use .jpg or .webp")

            # Encode frame
            ret, buffer = cv2.imencode(format, frame, encode_param)
            if not ret:
                return None

            return buffer.tobytes()
    def release(self):
        self._camera.release()
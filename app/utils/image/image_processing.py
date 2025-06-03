from typing import Tuple, Literal
from io import BytesIO
import numpy as np
import imageio, cv2, subprocess

class ImageProcessing:
    @staticmethod
    def crop_centre_frame(frame :np.ndarray,
                          size :Tuple[int,int]) -> np.ndarray:
        """
        Function for cropping centre frame
        :param frame:
        :param size: Size of rectangle box (x,y)
        :return:
        """
        # Shape
        height, width, _ = frame.shape
        # Centre point (x,y)
        centre_point = (int(width/2), int(height/2))
        # Start point (x,y)
        start_point = (int(centre_point[0] - size[0]/2), int(centre_point[1] - size[1]/2))
        return frame[start_point[1]:start_point[1] + size[1], start_point[0]:start_point[0] + size[0]]

    @staticmethod
    def convert_images_to_video_buffer(frames: list[np.ndarray],
                                       fps: int = 24,
                                       codec: str = 'libx264') -> BytesIO:
        buffer = BytesIO()
        # imageio will automatically choose ffmpeg if available
        with imageio.get_writer(buffer, format='mp4', mode = 'I', fps = fps, codec = codec) as writer:
            for frame in frames:
                writer.append_data(frame)
        buffer.seek(0)  # Reset the buffer position to the start
        return buffer

    @staticmethod
    def write_images_to_video(images :list[np.ndarray],
                              output_path :str,
                              backends :Literal["opencv","ffmpeg"] = "opencv",
                              fps :int = 30,
                              crf :int = 23,
                              preset = "medium",
                              codec = 'mp4v'):
        """
        Convert a list of image arrays to a video file using OpenCV.

        Parameters:
        - image_arrays: List of NumPy arrays (H, W, C) in RGB format
        - output_path: Path to save the output video (e.g., 'output.mp4')
        - fps (int): Frames per second.
        - crf (int): Constant Rate Factor (lower = better quality, typical: 18–28).
        - preset (str): Compression preset (ultrafast, superfast, medium, slow, etc.).
        """
        # Get dimensions from the first image
        height, width, channels = images[0].shape

        # Using opencv as backend
        if backends == "opencv":
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Write each frame
            for img in images:
                # Ensure image is in BGR format (OpenCV uses BGR)
                if img.shape[2] == 3:  # Assuming RGB input
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img
                writer.write(img_bgr)

            # Release the writer
            writer.release()
            return
        else:
            # Using ffmpeg as backend
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "-",
                "-an",  # No audio
                "-vcodec", "libx264",
                "-crf", str(crf),
                "-preset", preset,
                "-movflags", "+faststart",  # for web compatibility
                output_path
            ]
            # Init Popen
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            # Write
            for img in images: process.stdin.write(img.astype(np.uint8).tobytes())

            process.stdin.close()
            process.wait()
            return

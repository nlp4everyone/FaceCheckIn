from typing import Tuple
from io import BytesIO
import numpy as np
import imageio

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
    def images_to_video_buffer(frames: list[np.ndarray],
                               fps: int = 24,
                               codec: str = 'libx264') -> BytesIO:
        buffer = BytesIO()
        # imageio will automatically choose ffmpeg if available
        with imageio.get_writer(buffer, format='mp4', mode = 'I', fps = fps, codec = codec) as writer:
            for frame in frames:
                writer.append_data(frame)

        buffer.seek(0)  # Reset the buffer position to the start
        return buffer
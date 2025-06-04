# Minio component
from minio import Minio
from minio.datatypes import Bucket
# Other component
from minio.error import S3Error
from minio.helpers import ObjectWriteResult
# Typing
from typing import List, Union
# Other component
import numpy as np
import io
# Image
from app.utils.image import ImagePreprocess, ImageProcessing

class MinioObjectStorage:
    def __init__(self,
                 endpoint: str,
                 access_key: str | None = None,
                 secret_key: str | None = None,
                 **kwargs):
        # Init service
        self._minio_service = Minio(endpoint = endpoint,
                                    access_key = access_key,
                                    secret_key = secret_key,
                                    secure = False,
                                    **kwargs)

    @property
    def list_buckets(self)-> List[Bucket]:
        return self._minio_service.list_buckets()

    def file_exists(self,
                    bucket_name: str,
                    obj_name :str):
        """Check whether object existed in bucket or not"""
        try:
            self._minio_service.stat_object(bucket_name,
                                            obj_name)
            return True
        except S3Error as err:
            return False

    def create_bucket(self,
                      bucket_name :str,
                      **kwargs):
        # Create bucket while not existed
        if not self._minio_service.bucket_exists(bucket_name):
            self._minio_service.make_bucket(bucket_name = bucket_name,
                                            **kwargs)

    def upload_image(self,
                     bucket_name: str,
                     image :Union[np.ndarray,io.BytesIO],
                     image_name :str,
                     **kwargs) -> Union[ObjectWriteResult,None]:
        """Upload the image to MinIO"""

        # When file existed!
        if self.file_exists(bucket_name = bucket_name,
                            obj_name = image_name):
            return None
        # Convert to bytes if image is numpy array
        if isinstance(image, np.ndarray):
            # Convert image under numpy to bytes
            image = ImagePreprocess.convert_image_to_bytes(image)
        # Upload to bucket
        result = self._minio_service.put_object(bucket_name = bucket_name,
                                                object_name = image_name,
                                                data = image,
                                                length = len(image.getvalue()) ,
                                                content_type = "image/jpeg",
                                                **kwargs)
        return result

    def upload_video(self,
                     bucket_name: str,
                     images :List[np.ndarray],
                     video_name :str,
                     **kwargs) -> Union[ObjectWriteResult,None]:
        """Upload the image to MinIO"""

        # When file existed!
        if self.file_exists(bucket_name = bucket_name,
                            obj_name = video_name):
            return None

        # Numpy image case
        if not isinstance(images[0], np.ndarray):
            raise ValueError(f"Image must be list of numpy array, not {type(images[0])}")

        # Convert BGR image to RGB
        rgb_images = ImagePreprocess.convert_bgr_to_rgb(images)
        # Convert list of numpy to video buffer
        ImageProcessing.write_images_to_video(rgb_images,
                                              output_path = "temp.mp4",
                                              backends = "ffmpeg")

        # Read video as buffer
        with open("temp.mp4", 'rb') as f:
            buffer = f.read()

        byte_stream = io.BytesIO(buffer)

        # Upload to bucket
        result = self._minio_service.put_object(bucket_name = bucket_name,
                                                object_name = video_name,
                                                data = byte_stream,
                                                content_type = "video/mp4",
                                                length = byte_stream.getbuffer().nbytes,
                                                **kwargs)
        return result

    def remove_image(self,
                     image_name :str,
                     bucket_name :str):
        """Check image existed before removing"""
        if self.file_exists(bucket_name = bucket_name,
                            obj_name = image_name):
            # Delete the object
            try:
                self._minio_service.remove_object(bucket_name,
                                                  image_name)
            except S3Error as err:
                print("Error occurred while deleting object:", err)
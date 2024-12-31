from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
)
import numpy as np
import cv2
import tensorflow as tf

class GradCAM_Tools:

    @staticmethod
    class display:

        def GradCAM(model: tf.keras.models.Model, img: np.array) -> None:
            """
            Displays the GradCAM of a model given an image

            Used to interpret the focus of a conv layer

            Use 0 key to exit display
            """

            # Normalize Image
            img /= np.max(img)

            # Init GradCAM
            with GradCAM(model=model, target_layers=[model.get_layer('last-conv-layer')]) as cam:

                grayscale_cam = cam(input_tensor=img)
                
                visualization = show_cam_on_image(img * 255, grayscale_cam, use_rgb=True)


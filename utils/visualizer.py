import matplotlib.pyplot as plt, numpy as np
import tensorflow as tf
import keras
import cv2

from typing import List, Callable, Dict, Literal

from utils.featurizer import featurizer

class visualizer():

    @staticmethod
    def plot_fit(image: np.array, linear_color: str = 'red', poly_color: str = 'orange') -> None:

        linear_coefficents: List[float] = featurizer.getView(image)

        poly_coefficents: List[float] = featurizer.getViewPoly(image)

        plt.imshow(image, cmap='gray')

        x = np.linspace(0, image[0].__len__(), 1000)

        linear_var = np.array([x ** exp for exp in range(linear_coefficents.__len__())][::-1])

        poly_var = np.array([x ** exp for exp in range(poly_coefficents.__len__())][::-1])

        linear_y = np.dot(linear_var.T, np.array(linear_coefficents).T)

        poly_y = np.dot(poly_var.T, np.array(poly_coefficents).T)

        plt.plot(x, poly_y, color=poly_color)

        plt.plot(x, linear_y, color=poly_color)

    @staticmethod
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        """
        From Keras Docs -> https://keras.io/examples/vision/grad_cam/
        """

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()


    @staticmethod
    def grad_cam(img: np.array, model, color: Literal['red', 'green', 'blue'] = 'green', dim_factor: int = 2, last_conv_layer: str = 'last-conv-layer') -> plt:
        """
        color: Color for gradcam overlay
        dim_factor: How much to dim the background Mammography by using division
        """

        # Set color of gradcam heatmap
        def set_channels_green(gradcam_heatmap):
            gradcam_heatmap[:, :, :, [0, 2]] = 0

        def set_channels_red(gradcam_heatmap):
            gradcam_heatmap[:, :, :, [1, 2]] = 0

        def set_channels_blue(gradcam_heatmap):
            gradcam_heatmap[:, :, :, [0, 1]] = 0

        def set_channels_orange(gradcam_heatmap):

            # Green in half
            gradcam_heatmap[:, :, :, 1] /= 2
            
            # Blue to zero
            gradcam_heatmap[:, :, :, 2] = 0

        color_to_channels: Dict[str, Callable] = {
            'green': set_channels_green,
            'red': set_channels_red,
            'blue': set_channels_blue,
            'orange': set_channels_orange,
        }

        # Add batch dim to image

        img  = np.expand_dims(img, axis=0)

        # Get Heatmap

        gradcam_heatmap = np.tile(
            np.expand_dims(
                cv2.resize(
                    visualizer.make_gradcam_heatmap(img, model, last_conv_layer_name='last-conv-layer'), img.shape[1:3])
                , axis=-1)
            , (1,1,1,3))

        # Dim Mammography and change overlay color

        display_image = img / dim_factor

        color_to_channels[color](gradcam_heatmap)

        return plt.imshow((display_image + gradcam_heatmap).reshape((224,224,3)))


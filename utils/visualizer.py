import matplotlib.pyplot as plt, numpy as np

from typing import List

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
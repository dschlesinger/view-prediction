import matplotlib.pyplot as plt, numpy as np

from typing import List

class visualizer():

    @staticmethod
    def plot_fit(image: np.array, coefficents: List[float], color: str='red') -> None:

        plt.imshow(image, cmap='gray')

        x = np.linspace(0, image[0].__len__(), 1000)

        eval_variable = np.array([x ** exp for exp in range(coefficents.__len__())][::-1])

        y = np.dot(eval_variable.T, np.array(coefficents).T)

        plt.plot(x, y, color=color)
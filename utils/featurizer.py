import numpy as np, math

from typing import Tuple

class featurizer():

    @staticmethod
    def getLaterality(img: np.array) -> Tuple[float, float]:
        """
        Splits Midline and sum of either side
        """

        midline: int = round(img[0].__len__()/2)

        L_score = img[:, :midline].sum()

        R_score = img[:, midline:].sum()

        norm = L_score + R_score

        return L_score / norm, R_score / norm
    
    @staticmethod    
    def median(col: list, org_len: int) -> float:

        if col.__len__() < org_len / 16:

            return None

        elif col.__len__() % 2 == 0:
            return math.floor((col[int(col.__len__()/2)] + col[int((col.__len__()/2)-1)]) / 2)
        else:
            return col[math.floor(col.__len__()/2)]
        
    @staticmethod
    def getView(img: np.array) -> Tuple[float, float]: # Slope, Intercept
        """
        Finds Median of Bright Pixels then Finds Slope of Best Fit Line

        Returns (Slope, Intercept)
        """

        coords = [(i, featurizer.median(np.where(col > col.max()/4)[0], col.__len__())) for i, col in enumerate(img.T)]

        return np.polyfit([x for x,y in coords if y != None], [y for x,y in coords if y != None], 1)
    
    @staticmethod
    def getFarthestPoint(img, laterality) -> int: # returns row of farthest, lat true is left

        if laterality: # left

            farthest_nonzero_indices = [np.max(np.where(row != 0)[0]) if np.any(row != 0) else -1 for row in img]

            # Step 2: Find the row with the largest farthest value
            return np.argmax(farthest_nonzero_indices)

        else:

            farthest_nonzero_indices = [np.max(np.where(row != 0)[0]) if np.any(row != 0) else -1 for row in img[::-1]]

            # Step 2: Find the row with the largest farthest value
            return np.argmax(farthest_nonzero_indices)
        
    @staticmethod
    def getMirror(img: np.array, laterality: bool) -> Tuple[float]:

        midline = featurizer.getFarthestPoint(img, laterality)

        if midline <= img.__len__() / 2:

            tophalf = img[:midline, :]

            bottomhalf = img[midline:midline+tophalf.__len__(), :][:, ::-1]

        else:

            bottomhalf = img[midline:, :][:, ::-1]

            tophalf = img[midline-bottomhalf.__len__():midline, :]

        return np.abs(tophalf-bottomhalf).sum()

    @staticmethod    
    def getViewPoly(img: np.array) -> Tuple[float, float]: # Slope, Intercept
        """
        Finds Median of Bright Pixels then Finds Slope of Best Fit Line

        Returns (Slope, Intercept)
        """

        coords = [(i, featurizer.median(np.where(col > col.max()/4)[0], col.__len__())) for i, col in enumerate(img.T)]

        return np.polyfit([x for x,y in coords if y != None], [y for x,y in coords if y != None], 3)
    
    @staticmethod
    def getLaterality_parallel(imgs: np.array) -> Tuple[float, float]:
        """
        Splits Midline and sum of either side

        return left, right
        """

        midline = np.array([round(img[0].__len__()/2) for img in imgs])

        L_score = np.array([img[:, :ml].sum() for ml, img in zip(midline, imgs)])

        R_score = np.array([img[:, ml:].sum() for ml, img in zip(midline, imgs)])

        norm = L_score + R_score

        return L_score / norm, R_score / norm

    @staticmethod
    def getView_parallel(imgs: np.array) -> Tuple[float, float]: # Slope, Intercept
        """
        Finds Median of Bright Pixels then Finds Slope of Best Fit Line

        Returns (Slope, Intercept), (x^3, x^2, x, Intercept)
        """

        coords = [[(i, featurizer.median(np.where(col > col.max()/4)[0], col.__len__())) for i, col in enumerate(img.T)] for img in imgs]

        return [np.polyfit([x for x,y in c if y != None], [y for x,y in c if y != None], 1) for c in coords], [np.polyfit([x for x,y in c if y != None], [y for x,y in c if y != None], 3) for c in coords]
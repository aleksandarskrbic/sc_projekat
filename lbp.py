from skimage import feature
import numpy as np


class LBP:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def compute(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.num_points, self.radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.num_points + 3), range=(0, self.num_points + 2))

        hist = hist.astype('float')
        hist /= (hist.sum() + eps)

        return hist

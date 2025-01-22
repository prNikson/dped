import cv2
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from scipy.stats import pearsonr
import numpy as np


img1 = cv2.imread('res_image1.jpg')
img2 = cv2.imread('res_image2.jpg')
assert img1.shape == img2.shape

height, width, _ = img1.shape
c = 0
counter = 0


for x in range(1, width):
    for y in range(1, height):
        if counter >= 40000:
            break
        if x + 100 <= width and y + 100 <= height:
            patch1 = img1[x:x+100, y:y+100]
            patch2 = img2[x:x+100, y:y+100]
            patch1_copy = patch1.copy()
            patch2_copy = patch2.copy()
            patch1_copy = cv2.cvtColor(patch1_copy, cv2.COLOR_BGR2GRAY)
            patch2_copy = cv2.cvtColor(patch2_copy, cv2.COLOR_BGR2GRAY)
            flat1 = patch1_copy.flatten()
            flat2 = patch2_copy.flatten()
            correlation, _ = pearsonr(flat1, flat2)
            if correlation >= 0.85:
                c += 1
                cv2.imwrite(f'patches/sony/{c}.jpg', patch1)
                cv2.imwrite(f'patches/kvadra/{c}.jpg', patch2)
            # correlation = correlate2d(patch1, patch2, mode='full')
            # max_cor = np.mean(correlation)
            # norm_correlation = max_cor / (np.linalg.norm(patch1) * np.linalg.norm(patch2))
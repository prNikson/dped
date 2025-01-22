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
# cor = np.array([])

for x in range(1, width):
    for y in range(1, height):
        if counter >= 40000:
            break
        if x + 99 <= width and y + 99 <= height:
            patch1 = img1[x:x+99, y:y+99]
            patch2 = img2[x:x+99, y:y+99]
            
            patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
            patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)
            flat1 = patch1.flatten()
            flat2 = patch2.flatten()
            correlation, _ = pearsonr(flat1, flat2)
            if correlation >= 0.85:
                c += 1
            counter += 1
            # correlation = correlate2d(patch1, patch2, mode='full')
            # max_cor = np.mean(correlation)
            # norm_correlation = max_cor / (np.linalg.norm(patch1) * np.linalg.norm(patch2))
            # plt.figure(figsize=(12, 6))

            # plt.subplot(1, 3, 1)
            # plt.title('Image 1')
            # plt.imshow(patch1, cmap='gray')
            # plt.axis('off')

            # plt.subplot(1, 3, 2)
            # plt.title('Image 2')
            # plt.imshow(patch2, cmap='gray')
            # plt.axis('off')

            # plt.subplot(1, 3, 3)
            # plt.title('Correlation')
            # plt.imshow(correlation, cmap='hot')
            # plt.axis('off')

            # plt.show()
print(c)
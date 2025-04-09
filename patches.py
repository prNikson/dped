import cv2
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from scipy.stats import pearsonr
import numpy as np


img1 = cv2.imread('res_1.jpg')
img2 = cv2.imread('res_2.jpg')

assert img1.shape == img2.shape

height, width, _ = img1.shape
c = 0
counter = 0

patch_arr = list()
cor_arr = list()

x1, y1 = 0, 0
for y in range(1, height, 100):
    for x in range(1, width, 100):
        if x + 109 < width and y + 109 < height:
            start_x = 1
            finish_x = width
            start_y = 1
            finish_y = height

            kx = 0
            ky = 0

            if x - start_x <= 10:
                kx = x - start_x
            elif finish_x - x <= 10:
                kx = finish_x - x
            else:
                kx = 10
                x1 = x - 10
            
            if y - start_y <= 10:
                ky = y - start_y
            elif finish_y - y <= 10:
                ky = finish_y - y
            else:
                ky = 10
                y1 = y - 10
            
            patch1 = img1[y:y+100, x:x+100]
            patch1_copy = patch1.copy()
            for j in range(1, 11 + ky):
                for i in range(1, 11 + kx):
                    patch2 = img2[y1+j:y1+100+j, x1+i:x1+100+i]
                    patch2_copy = patch2.copy()
                    # print(f'{x1+i}:{x1+100+i}')
                    flat1 = cv2.cvtColor(patch1_copy, cv2.COLOR_BGR2GRAY).flatten()
                    flat2 = cv2.cvtColor(patch2_copy, cv2.COLOR_BGR2GRAY).flatten()
                    correlation, _ = pearsonr(flat1, flat2)
                    patch_arr.append(patch2)
                    cor_arr.append(correlation)
            max_cor = max(cor_arr)
            if max_cor >= 0.8:
                i = cor_arr.index(max_cor)
                cv2.imwrite(f'patches/sony/{c}.jpg', patch1)
                cv2.imwrite(f'patches/kvadra/{c}.jpg', patch_arr[i])
                c += 1
            cor_arr = []
            patch_arr = []
            # correlation = correlate2d(patch1, patch2, mode='full')
            # max_cor = np.mean(correlation)
            # norm_correlation = max_cor / (np.linalg.norm(patch1) * np.linalg.norm(patch2))
        
    
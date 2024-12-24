import cv2
import numpy as np
import matplotlib.pyplot as plt

img2 = cv2.imread('pairs/1/camera.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('pairs/1/kvadra.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Используем BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
print(min(points1)[0])
# # Найдите матрицу преобразования
# M, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

# # Примените преобразование к одному из изображений
# h, w = img1.shape
# img1_transformed = cv2.warpPerspective(img1, M, (w, h))
# img2_resized = cv2.resize(img2, (img1_transformed.shape[1], img1_transformed.shape[0]))
# # Найдите пересечение
# intersection = cv2.bitwise_and(img1_transformed, img2_resized)

# cv2.imwrite('intersection.jpg', intersection)
# cv2.imshow('Intersection', intersection)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

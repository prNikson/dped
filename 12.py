import cv2
import numpy as np

path = '/media/sergey/DATA'
img1 = cv2.imread(path + '/B/JPEG/000373.jpg')     # Фото с камеры
img2 = cv2.imread(path + '/A/JPEG/000373.jpg')     # Фото с планшета

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.AKAZE_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

good_matches = matches[:50]

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

height, width = img1.shape[:2]
warped_img2 = cv2.warpPerspective(img2, H, (width, height))

overlap_mask = (cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) > 0) & (cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) > 0)

if not (np.count_nonzero(overlap_mask) == 0):
    ys, xs = np.where(overlap_mask)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    crop1 = img1[y_min:y_max, x_min:x_max]
    crop2 = warped_img2[y_min:y_max, x_min:x_max]

    cv2.imwrite('crop_camera.jpg', crop1)
    cv2.imwrite('crop_tablet.jpg', crop2)
import cv2
import numpy as np
import matplotlib.pyplot as plt


class CutIntersection:

    def __init__(self, camera_image: str, kvadra_image: str):
        self.image1 = cv2.imread(camera_image)
        self.image2 = cv2.imread(kvadra_image)
    
    def __crop_img(self, img, scale=0.1):
        center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
        width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
        left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
        top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
        img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
        return img_cropped

    def __homography(self):
        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()

        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Сопоставляем дескрипторы
        matches = bf.match(descriptors1, descriptors2)

        # Сортируем совпадения по расстоянию
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    
    def find_area(self):
        self.image1 = self.__crop_img(self.image1, 0.85)
        self.image2 = self.__crop_img(self.image2, 0.86)
        
        H = self.__homography()

        # transform_image1 = cv2.warpPerspective(self.image1, H, (self.image2.shape[1], self.image2.shape[0]))
        self.transform_image1 = self.image1

        h, w, _ = self.image1.shape

        corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype='float32')

        transform_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)

        inv_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)

        mask_image2 = np.zeros_like(self.image2)
        cv2.fillConvexPoly(mask_image2, np.int32(transform_corners), (255, 255, 255))

        self.transform_image2 = cv2.bitwise_and(self.image2, mask_image2)

        # transform_image1 = cv2.warpPerspective(transform_image1, np.linalg.inv(H), (w, h))
        self.transform_image2 = cv2.warpPerspective(self.transform_image2, np.linalg.inv(H), (w, h))

        self.show_result()    

    def show_result(self):
        plt.figure(figsize=(15, 10))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB))
        plt.title('Image 2')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(self.transform_image1, cv2.COLOR_BGR2RGB))
        plt.title('Image 1')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(self.transform_image2, cv2.COLOR_BGR2RGB))
        plt.title('Transform Image 2')
        plt.axis('off')

        plt.show()

CutIntersection('pairs/5/camera.jpg', 'pairs/5/kvadra.jpg').find_area()
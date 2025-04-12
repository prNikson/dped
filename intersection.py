import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class CutIntersection:

    def __init__(
            self,
            camera_image: str,
            kvadra_image: str,
            type: int,
            scale: float | None
            ):

        self.image1 = cv2.imread(camera_image)
        self.image2 = cv2.imread(kvadra_image)
        self.filename = Path(camera_image).stem
        self.type = type
        self.scale = float(scale) if scale is not None else None
    
    def __crop_img(self, img, scale):
        center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
        width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
        left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
        top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
        img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
        return img_cropped

    def __homography(self, img1, img2):

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create() if self.type == hash('orb') else cv2.SIFT_create()

        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = bf.match(descriptors1, descriptors2)

        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # self.show_img(cv2.drawMatches(self.image1, keypoints1, self.image2, keypoints2, matches[:15], outImg = None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
        return H
    
    def find_area(self):
        # self.image1 = self.__crop_img(self.image1, 0.85)
        # self.image2 = self.__crop_img(self.image2, 0.86)
        H = self.__homography(self.image1, self.image2)
        h, w, _ = self.image1.shape
        ht, wt, _ = self.image2.shape
        self.transform_image1 = cv2.warpPerspective(self.image1, H, (wt, ht))
        # self.transform_image1 = self.image1
        self.corners = self.create_corners(w, h)

        self.transform_corners = cv2.perspectiveTransform(self.corners.reshape(-1, 1, 2), H).reshape(-1, 2)
        self.inv_corners = cv2.perspectiveTransform(self.transform_corners.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
        self.inv_corners_int = np.int32(self.inv_corners)

        t = self.transform_corners
        x0 = int(max(t[0][0], t[1][0]))
        x1 = int(min(t[2][0], t[3][0]))
        y0 = int(max(t[0][1], t[3][1]))
        y1 = int(min(t[1][1], t[2][1]))
        corners = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
        mask_image2 = np.zeros_like(self.image2)
        cv2.fillConvexPoly(mask_image2, np.int32(self.transform_corners), (255, 255, 255))
        self.transform_image2 = cv2.bitwise_and(self.image2, mask_image2)
        # self.transform_image2 = self.find_mask(self.transform_image2)


        self.transform_image1 = cv2.warpPerspective(self.transform_image1, np.linalg.inv(H), (wt, ht))
        self.transform_image2 = cv2.warpPerspective(self.transform_image2, np.linalg.inv(H), (wt, ht))
        print(self.transform_image1.dtype)
        self.show_result()
        if self.scale is not None:
            self.crop_img(self.scale)
        cv2.imwrite(f'pairs/{self.filename}/res_1.jpg', self.transform_image1)
        cv2.imwrite(f'pairs/{self.filename}/res_2.jpg', self.transform_image2)

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

    def find_black_area(self):
        h, w, _ = self.transform_image2.shape
        corners = self.create_corners(w, h)
        s1, p1 = corners[0], self.inv_corners_int[0]
        s2, p2 = corners[1], self.inv_corners_int[1]
        s3, p3 = corners[2], self.inv_corners_int[2]
        s4, p4 = corners[3], self.inv_corners_int[3]

        if p1[0] > s1[0]:
            self.transform_image1 = self.transform_image1[:, p1[0]:]
            self.transform_image2 = self.transform_image2[:, p1[0]:]
            print(1)
        if p1[1] > s1[1]:
            self.transform_image1 = self.transform_image1[p1[1]:, :]
            self.transform_image2 = self.transform_image2[p1[1]:, :]
            print(2)
        if p2[0] > s2[0]:
            self.transform_image1 = self.transform_image1[:, p2[0]:]
            self.transform_image2 = self.transform_image2[:, p2[0]:]
            print(3)
        if p2[1] < s2[1]:
            self.transform_image1 = self.transform_image1[:p2[1], :]
            self.transform_image2 = self.transform_image2[:p2[1], :]
            print(4)
        if p3[0] < s3[0]:
            self.transform_image1 = self.transform_image1[:, :p3[0]]
            self.transform_image2 = self.transform_image2[:, :p3[0]]
            print(5)
        if p3[1] < s3[1]:
            self.transform_image1 = self.transform_image1[:p3[1], :]
            self.transform_image2 = self.transform_image2[:p3[1], :]
            print(6)
        if p4[0] < s4[0]:
            self.transform_image1 = self.transform_image1[:, :p4[0]]
            self.transform_image2 = self.transform_image2[:, :p4[0]]
            print(7)
        if p4[1] > s4[1]:
            self.transform_image1 = self.transform_image1[p4[1]:, :]
            self.transform_image2 = self.transform_image2[p4[1]:, :]
            print(8)

    def show_img(self, img):
        plt.subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def crop_img(self, scale=0.9):
        self.transform_image1 = (self.__crop_img(self.find_mask(self.transform_image1), scale))
        self.transform_image2 = (self.__crop_img(self.find_mask(self.transform_image2), scale))
        self.show_result()

    def find_mask(self, image):
        mask = np.all(image != [0, 0, 0], axis=-1).astype(np.uint8) * 255
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]

    def remove_px(self, px):
        h, w = self.transform_image1.shape[:2]
        self.transform_image1 = self.transform_image1[px:h-px, px:w-px]
        self.transform_image2 = self.transform_image2[px:h-px, px:w-px]

    def create_corners(self, w, h):
        return np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype='float32')
import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def gomography(image1, image2) -> list:
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

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

def area(image1, image2):
    image1 = crop_img(image1, 0.85)
    image2 = crop_img(image2, 0.86)
    H = gomography(image1, image2)
    transform_image1 = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))
    h, w, _ = image1.shape
    corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype='float32')
    transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    mask_image2 = np.zeros_like(image2)
    cv2.fillConvexPoly(mask_image2, np.int32(transformed_corners), (255, 255, 255))
    transform_image2 = cv2.bitwise_and(image2, mask_image2)
    transform_image1 = cv2.warpPerspective(transform_image1, np.linalg.inv(H), (image1.shape[1], image1.shape[0]))
    transform_image2 = cv2.warpPerspective(transform_image2, np.linalg.inv(H), (image1.shape[1], image1.shape[0]))
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(transform_image1, cv2.COLOR_BGR2RGB))
    plt.title('Highlighted Area in Image 2')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(transform_image2, cv2.COLOR_BGR2RGB))
    plt.title('Highlighted Area in Image 2')
    plt.axis('off')
    plt.show()
    cv2.imwrite('res_image1.jpg', transform_image1)
    cv2.imwrite('res_image2.jpg', transform_image2)

image1 = cv2.imread('pairs/2/camera.jpg')
image2 = cv2.imread('pairs/2/kvadra.jpg')

result_image = area(image1, image2)
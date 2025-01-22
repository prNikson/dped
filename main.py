import cv2
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
# import torch

# from transformers import AutoImageProcessor, SuperPointForKeypointDetection

# processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
# model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

# @torch.no_grad
def highlight_matching_area(image1, image2):
    image1 = crop_img(image1, 0.7)
    image2 = crop_img(image2, 0.9)
    # Преобразуем изображения в оттенки серого
    # image1 = crop_img(image1, 0.8)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # print(image1.shape, image2.shape)
    # Инициализируем SIFT детектор
    sift = cv2.ORB_create()

    # Находим ключевые точки и дескрипторы
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # images = [image1, image2]
    # images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
    # image_sizes = [(img.size[1], img.size[0]) for img in images]

    # inputs = processor(images, return_tensors="pt")
    # outputs = model(**inputs)
    # outputs = processor.post_process_keypoint_detection(outputs, image_sizes)

    # keypoints1, keypoints2 = [[cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in outputs[i]['keypoints'].float().numpy()] for i in range(2)]
    # descriptors1, descriptors2 = [outputs[i]['descriptors'].numpy() for i in range(2)]

    # Создаем объект для сопоставления дескрипторов
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Сопоставляем дескрипторы
    matches = bf.match(descriptors1, descriptors2)

    # Сортируем совпадения по расстоянию
    matches = sorted(matches, key=lambda x: x.distance)


    # Выделяем угловые точки, которые показывают пересекающуюся область
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # mx = 0
    # mx2 = 0
    # for mt in matches:
    #     p1 = keypoints1[mt.queryIdx].pt
    #     p2 = keypoints2[mt.trainIdx].pt
    #     if p1[0] > mx:
    #         if image1.shape[1] - p1[0] < 100 and image2.shape[1] - p2[0] < 100:
    #             x1, x2 = p1, p2
    #             mx = p1[0]
    #     if p2[0] > mx2:
    #         if image1.shape[1] - p1[0] < 100 and image2.shape[1] - p2[0] < 100:
    #             y1, y2 = p1, p2
    #             mx2 = p2[0]

    # cv2.circle(image1, (int(x1[0]), int(x1[1])), 10, (0, 255, 0), 2)
    # cv2.circle(image2, (int(x2[0]), int(x2[1])), 10, (0, 255, 0), 2)
    # cv2.circle(image1, (int(y1[0]), int(y1[1])), 10, (255, 0, 0), 2)
    # cv2.circle(image2, (int(y2[0]), int(y2[1])), 10, (255, 0, 0), 2)

    # Находим гомографию
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # Извлекаем угловые точки из первого изображения
    h, w = image2.shape[:2]
    transform_image1 = cv2.warpPerspective(image1, M, (w, h))
    h, w, _ = image1.shape
    # corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype='float32')
    # transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)

    # Найдите границы общей области
    # x_min = int(min(transformed_corners[:, 0]))
    # x_max = int(max(transformed_corners[:, 0]))
    # y_min = int(min(transformed_corners[:, 1]))
    # y_max = int(max(transformed_corners[:, 1]))
    # x_min = x_min if x_min >= 0 else 0
    # x_max = x_max if x_max >= 0 else 0
    # y_min = y_min if y_min >= 0 else 0
    # y_max = y_max if y_max >= 0 else 0
    # Вырежьте общую область из второго изображения
    # common_area = image2[y_min:y_max, x_min:x_max]
    corners_image1 = np.array([[0, 0], [0, image1.shape[0] - 1], [image1.shape[1] - 1, image1.shape[0] - 1], [image1.shape[1] - 1, 0]], dtype='float32').reshape(-1, 1, 2)
    corners_image2 = cv2.perspectiveTransform(corners_image1, M)



    # corners_image2[2][0][0] = x2[0]
    # corners_image2 = np.append(corners_image2[0][:2], [[x1[0], image1.shape[0] - 1], [x1[0], 0]])
    # corners_image2 = corners_image1
    # Создаем маску для выделения области
    mask_image2 = np.zeros_like(image2)
    cv2.fillConvexPoly(mask_image2, np.int32(corners_image2), (255, 255, 255))

    # # # Применяем маску к изображению
    result_image = cv2.bitwise_and(image2, mask_image2)
    # result_image = cv2.warpPerspective(result_image, np.linalg.inv(M), (image1.shape[1], image1.shape[0]))
    # transform_image1 = cv2.warpPerspective(transform_image1, np.linalg.inv(M), (image1.shape[1], image1.shape[0]))

    # res_template = cv2.matchTemplate(result_image, image1, cv2.TM_CCOEFF)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_template)
    # top_left = min_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(image1,top_left, bottom_right, 255, 2)
    corners_image2_int = np.int32(corners_image2)  # Преобразуем в целые числа
    cv2.polylines(image2, [corners_image2_int], isClosed=True, color=(0, 255, 0), thickness=3)
    # Отображаем результаты
    plt.figure(figsize=(15, 10))
    # print(image1.shape, image2.shape)
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    # plt.imshow(cv2.cvtColor(cv2.warpPerspective(transform_image2, np.linalg.inv(M), (image1.shape[1], image1.shape[0])), cv2.COLOR_BGR2RGB))
    # plt.imshow(cv2.cvtColor(common_area, cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(transform_image1, cv2.COLOR_BGR2RGB))
    plt.title('Highlighted Area in Image 2')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Highlighted Area in Image 2')
    plt.axis('off')

    plt.show()

    cv2.imwrite('res_image1.jpg', transform_image1)
    cv2.imwrite('res_image2.jpg', result_image)



# Пример использования функции
image1 = cv2.imread('pairs/11/camera.jpg')
image2 = cv2.imread('pairs/11/kvadra.jpg')
highlight_matching_area(image1, image2)

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

def highlight_matching_area(image1, image2):
    # Преобразуем изображения в оттенки серого
    # image1 = crop_img(image1, 0.8)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # print(image1.shape, image2.shape)
    # Инициализируем SIFT детектор
    sift = cv2.SIFT_create()

    # Находим ключевые точки и дескрипторы
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Создаем объект для сопоставления дескрипторов
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Сопоставляем дескрипторы
    matches = bf.match(descriptors1, descriptors2)

    # Сортируем совпадения по расстоянию
    matches = sorted(matches, key=lambda x: x.distance)
    

    # Выделяем угловые точки, которые показывают пересекающуюся область
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    mx = 0
    mx2 = 0
    for mt in matches:
        p1 = keypoints1[mt.queryIdx].pt
        p2 = keypoints2[mt.trainIdx].pt
        if p1[0] > mx:
            if image1.shape[1] - p1[0] < 100 and image2.shape[1] - p2[0] < 100:
                x1, x2 = p1, p2
                mx = p1[0]
        if p2[0] > mx2:
            if image1.shape[1] - p1[0] < 100 and image2.shape[1] - p2[0] < 100:
                y1, y2 = p1, p2
                mx2 = p2[0]

    # cv2.circle(image1, (int(x1[0]), int(x1[1])), 10, (0, 255, 0), 2)
    # cv2.circle(image2, (int(x2[0]), int(x2[1])), 10, (0, 255, 0), 2)
    # cv2.circle(image1, (int(y1[0]), int(y1[1])), 10, (255, 0, 0), 2)
    # cv2.circle(image2, (int(y2[0]), int(y2[1])), 10, (255, 0, 0), 2)

    # Находим гомографию
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    # Извлекаем угловые точки из первого изображения
    h, w = image2.shape[:2]
    transform_image2 = cv2.warpPerspective(image1, M, (w, h))
    corners_image1 = np.array([[0, 0], [0, image1.shape[0] - 1], [image1.shape[1] - 1, image1.shape[0] - 1], [image1.shape[1] - 1, 0]], dtype='float32').reshape(-1, 1, 2)

    corners_image2 = cv2.perspectiveTransform(corners_image1, M)
    corners_image2[2][0][0] = x2[0]
    # corners_image2 = np.append(corners_image2[0][:2], [[x1[0], image1.shape[0] - 1], [x1[0], 0]])
    # corners_image2 = corners_image1
    # Создаем маску для выделения области
    mask_image2 = np.zeros_like(image2)
    cv2.fillConvexPoly(mask_image2, np.int32(corners_image2), (255, 255, 255))

    # # # Применяем маску к изображению
    result_image = cv2.bitwise_and(image2, mask_image2)
    # result_image = cv2.warpPerspective(result_image, np.linalg.inv(M), (image1.shape[1], image1.shape[0]))
    # transform_image2 = cv2.warpPerspective(transform_image2, np.linalg.inv(M), (image1.shape[1], image1.shape[0]))
    # cv2.imwrite('res.jpg', result_image)
    # # Соединяем угловые точки в прямоугольник
    corners_image2_int = np.int32(corners_image2)  # Преобразуем в целые числа
    cv2.polylines(image2, [corners_image2_int], isClosed=True, color=(0, 255, 0), thickness=3)
    # Отображаем результаты
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 4, 1)
    # plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    # plt.imshow(cv2.cvtColor(cv2.warpPerspective(transform_image2, np.linalg.inv(M), (image1.shape[1], image1.shape[0])), cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(transform_image2, cv2.COLOR_BGR2RGB))
    plt.title('Highlighted Area in Image 2')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Highlighted Area in Image 2')
    plt.axis('off')

    plt.show()

    # cv2.imwrite('result.jpg', result_image)

# Пример использования функции
image1 = cv2.imread('pairs/11/camera.jpg')
image2 = cv2.imread('pairs/11/kvadra.jpg')
highlight_matching_area(image1, image2)
import cv2
import numpy as np
import matplotlib.pyplot as plt

def highlight_matching_area(image1, image2):
    # Преобразуем изображения в оттенки серого
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

    # Находим гомографию
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    # Извлекаем угловые точки из первого изображения
    h, w = image2.shape[:2]
    corners_image1 = np.array([[0, 0], [0, image1.shape[0] - 1], [image1.shape[1] - 1, image1.shape[0] - 1], [image1.shape[1] - 1, 0]], dtype='float32').reshape(-1, 1, 2)
    corners_image2 = cv2.perspectiveTransform(corners_image1, M)

    # Создаем маску для выделения области
    mask_image2 = np.zeros_like(image2)
    cv2.fillConvexPoly(mask_image2, np.int32(corners_image2), (255, 255, 255))

    # # # Применяем маску к изображению
    result_image = cv2.bitwise_and(image2, mask_image2)
    cv2.imwrite('res.jpg', result_image)
    # # Соединяем угловые точки в прямоугольник
    corners_image2_int = np.int32(corners_image2)  # Преобразуем в целые числа
    cv2.polylines(image2, [corners_image2_int], isClosed=True, color=(0, 255, 0), thickness=3)

    # Отображаем результаты
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2 with Rectangle')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Highlighted Area in Image 2')
    plt.axis('off')

    plt.show()

# Пример использования функции
image1 = cv2.imread('pairs/4/camera.jpg')
image2 = cv2.imread('pairs/4/kvadra.jpg')
highlight_matching_area(image1, image2)
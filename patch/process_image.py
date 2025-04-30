import cv2
from scipy.signal import correlate2d
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm, trange


def process_image(arg: tuple[str, int]) -> list | None:
    filename, index = arg
    if (res := intersection(filename)) is not None:
        img1, img2 = res

        if img1.shape != img2.shape:
            return None

        height, width, _ = img1.shape
        c = 0
        counter = 0
        cross_corr = 0.9

        patch_arr = list()
        cor_arr = list()
        patches_list = list()

        x1, y1 = 0, 0
        for y in trange(1, height, 100):
            print(f"************ {index + 1} step ****************")
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
                            flat1 = cv2.cvtColor(patch1_copy, cv2.COLOR_BGR2GRAY).flatten()
                            flat2 = cv2.cvtColor(patch2_copy, cv2.COLOR_BGR2GRAY).flatten()
                            correlation, _ = pearsonr(flat1, flat2)
                            patch_arr.append(patch2)
                            cor_arr.append(correlation)
                    max_cor = max(cor_arr)
                    if max_cor >= cross_corr:
                        i = cor_arr.index(max_cor)
                        patches_list.append(
                            [
                                np.transpose(patch1, (2, 0, 1)),
                                np.transpose(patch_arr[i], (2, 0, 1))
                            ]
                        )
                        c += 1
                    cor_arr = []
                    patch_arr = []
                # correlation = correlate2d(patch1, patch2, mode='full')
                # max_cor = np.mean(correlation)
                # norm_correlation = max_cor / (np.linalg.norm(patch1) * np.linalg.norm(patch2))
        return patches_list

def intersection(filename: str) -> tuple | None:

    img1 = cv2.imread(f'/home/miriteam/Desktop/B/JPEG/{filename}')
    img2 = cv2.imread(f'/home/miriteam/Desktop/A/JPEG/{filename}')

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
    dst_pts = np.float32([kp2[m.pathtrainIdx].pt for m in good_matches]).reshape(-1,1,2)

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

        return (crop1, crop2)
    return None
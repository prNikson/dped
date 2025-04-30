import argparse
from intersection import CutIntersection
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('img')
parser.add_argument('type', help='Type of keypoints')
parser.add_argument('-s', help='coeff: 0.9')
parser.add_argument('-t', help='second intersection')
if __name__ == "__main__":
	# img = parser.parse_args().img
	# type = hash(parser.parse_args().type)
	# scale = parser.parse_args().s
	# image_path = Path(img).stem
	# path = '/home/miriteam/Desktop/'
	# # a = CutIntersection(path + f'B/JPEG/{img}', path + f'A/JPEG/{img}', type, scale)
	# pair = '583'
	# a = CutIntersection(f'pairs/000{pair}/res_1.jpg', f'pairs/000{pair}/res_2.jpg', type, scale, '000' + pair)
	# a.find_area()
    # for i in trange(632, 745):
    #     try:
    #         a.find_area()
    #     except Exception as e:
    #         print(e)
    # for i in Path('pairs').glob('*'):
    #     img1 = cv2.imread(str(i) + '/res_1.jpg')
    #     img2 = cv2.imread(str(i) + '/res_2.jpg')

    #     if img1.shape != img2.shape:
    #         print(str(i))
    pass
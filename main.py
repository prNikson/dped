import argparse
from intersection import CutIntersection

parser = argparse.ArgumentParser()
parser.add_argument('img')
parser.add_argument('type', help='Type of keypoints')
parser.add_argument('-s', help='coeff: 0.9')


if __name__ == "__main__":
    img = parser.parse_args().img
    type = hash(parser.parse_args().type)
    scale = parser.parse_args().s
    print(img)
    path = '/media/sergey/DATA/'
    a = CutIntersection(path + f'B/JPEG/{img}', path + f'A/JPEG/{img}', type, scale)
    a.find_area()

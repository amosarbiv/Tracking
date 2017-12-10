import cv2

from utils.generate_binary_movie import frameGenerator
from utils.mouse import Selector
from utils.utils import readImages



def main():
    frames = frameGenerator(steps=3) if not args.images_path else readImages(args.images_path)
    frame = next(frames)[0]
    x, y, w, h, window_name = Selector(frame).accuireTarget()
    for frame, *_ in frames:
        cv2.imshow(window_name, frame)
        cv2.waitKey(0)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images-path')
    parser.add_argument('-s', '--frame-size',type=int, default=200)
    args = parser.parse_args()
    main()

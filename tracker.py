import cv2

from utils.mouse import Selector
from utils.utils import readImages
from utils.static_image.detection import Foreground



def main():
    # frameGenerator(steps=3) if not args.images_path else
    frames = readImages(args.images_path)
    frame = next(frames)[0]
    foreground = Foreground()
    x, y, w, h, window_name = Selector(frame).accuireTarget()
    color = foreground.hue_to_track(frame, x, y, w, h)
    prev2, prev = None, None
    for frame, *_ in frames:
        # frame = foreground.binary_foreground_(frame) # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.12.3705&rep=rep1&type=pdf
        # frame = foreground.binary_foreground(frame, black=True) # hsv for black
        # frame = foreground.binary_foreground(frame, hsv_to_track=color) # hsv by selection
        if prev2 is not None:
            frame = foreground.binary_foreground_by_movement(prev2, prev, frame)
        prev2 = prev
        prev = frame


        cv2.imshow(window_name, frame)
        cv2.waitKey(0)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images-path')
    parser.add_argument('-s', '--frame-size', type=int, default=200)
    args = parser.parse_args()
    main()

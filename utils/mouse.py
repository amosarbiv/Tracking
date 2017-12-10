import cv2
import os

class Selector():
    def __init__(self, image):
        if isinstance(image, str):
            assert(os.path.exists(image),True)
            self.image = cv2.imread(image)
        else:
            self.image = image

    def accuireTarget(self):
        window_name = "Image"
        y, x, h, w = [int(item) for item in cv2.selectROI(self.image)]
        rect = cv2.rectangle(self.image,(y, x), (y+h, x+w),(255,0,0),2)
        cv2.destroyAllWindows()
        cv2.imshow(window_name, rect)
        cv2.waitKey(0)
        return x, y, w, h, window_name


if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True)
    args = parser.parse_args()
    x, y, w, h, *_ = Selector(args.image).accuireTarget()
    print(x, y, w, h)
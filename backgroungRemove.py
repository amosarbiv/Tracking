import cv2
import numpy as np


class BackGroundSubtractor:
    def __init__(self, firstFrame):
        self.backGroundModel = denoise(firstFrame).astype(np.uint8)

    def getForeground(self, frame):
        res = cv2.absdiff(self.backGroundModel, frame)
        self.backGroundModel = frame.astype(np.uint8)
        return res

    def getMask(self, frame):
        foreGround = self.getForeground(denoise(frame))
        ret, mask = cv2.threshold(foreGround, 15, 255, cv2.THRESH_BINARY)
        return mask

    def get_binary(self, frame):
        foreGround = self.getForeground(denoise(frame))
        foreGround = cv2.cvtColor(foreGround, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(foreGround, 15, 255, cv2.THRESH_BINARY)
        return mask



def denoise(frame):
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame


def main():
    s = "/Users/orrbarkat/Downloads/imagedata++/06-MotionSmoothness/06-MotionSmoothness_video00007/00000"
    frame = cv2.imread(s + "004.jpg")
    backSubtractor = BackGroundSubtractor(denoise(frame))
    for i in range(4, 793):
        num = str(i)
        if (len(num) < 3):
            num = '0' * (3 - len(num)) + num

        frame = cv2.imread(s + num + ".jpg")
        mask = backSubtractor.getMask(frame)

        cv2.imshow('img', mask)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


if __name__ == "__main__":
    main()

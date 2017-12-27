import cv2
import numpy as np
from scipy import signal

class Foreground():
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    @staticmethod
    def binary_foreground(img, hsv_to_track=np.uint8([0, 0, 200]), black=False):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        if black:
            lower_hsv = np.array([0, 0, 0])
            upper_hsv = np.array([180, 255, 80])
        else:
            target_color_h = hsv_to_track[0]
            tolerance = 15
            lower_hsv = np.array([max(0, target_color_h - tolerance), 10, 10])
            upper_hsv = np.array([min(179, target_color_h + tolerance), 250, 250])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask)

        return res

    @staticmethod
    def binary_foreground_(img):
        # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html#background-subtraction
        return Foreground.fgbg.apply(img)

    @staticmethod
    def binary_foreground_by_movement(prev2, prev_frame, frame):
        prev2 = cv2.cvtColor(prev2, cv2.COLOR_BGR2GRAY)
        prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff1 = np.abs(prev2 - current)
        diff2 = np.abs(prev - current)
        mask1 = cv2.inRange(diff1, np.uint8([70]), np.uint8([255]))
        mask2 = cv2.inRange(diff2, np.uint8([70]), np.uint8([255]))
        mask = cv2.bitwise_and(mask1, mask1, mask=mask2)
        mask = Foreground.smooth_mask(mask)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        return res


    @staticmethod
    def hue_to_track(img, x, y, w, h):
        center_x = x + w//2
        center_y = y + h//2
        pixel = np.array([[img[center_y, center_x, :]]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV).squeeze()
        return hsv

    #still not working - this is pixel flow...
    @staticmethod
    def smooth_mask(mask, size=5, thresh=0.5):
        # smooth = np.copy(mask)
        filter = np.ones((size, size)) / 255.*size*2
        # for i in range(mask.shape[0] - size):
        #     for j in range(mask.shape[1] - size):
        #         count = np.sum(mask[i:i+5, j:j+5]) / 255
        #         percentage = count / size**2
        #         if percentage > thresh:
        #             smooth[i,j] = 255
        smooth = signal.convolve2d(mask, filter, 'same')
        smooth = cv2.inRange(smooth, 0.3, 1)
        return smooth



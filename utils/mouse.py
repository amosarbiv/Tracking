import cv2
import os

class Selector():
    def __init__(self, image):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image

    def accuireTarget(self):
        window_name = "Image"
        x, y, w, h = [int(item) for item in cv2.selectROI(self.image)]
        rect = cv2.rectangle(self.image,(x, y), (x+w, y+h),(255,0,0),2)
        cv2.destroyAllWindows()
        cv2.imshow(window_name, rect)
        cv2.waitKey(0)
        return x, y, w, h, window_name



import cv2
import numpy as np

class BackGroundSubtractor:
    def __init__(self,firstFrame):
        self.backGroundModel = firstFrame

    def getForeground(self,frame):
        backGroundModel =  frame + self.backGroundModel * (- 1)
        return cv2.absdiff(self.backGroundModel.astype(np.uint8),frame)


def denoise(frame):
    frame = cv2.medianBlur(frame,5)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    return frame

def main():
    s = "video//00000"
    frame = cv2.imread(s+"004.jpg")
    backSubtractor = BackGroundSubtractor(denoise(frame))
    for i in range(4,793):
        num = str(i)
        if (len(num) < 3):
                num = '0'*(3-len(num)) + num
       
        frame = cv2.imread(s+num+".jpg")

        foreGround = backSubtractor.getForeground(denoise(frame))
        ret, mask = cv2.threshold(foreGround, 15, 255, cv2.THRESH_BINARY)
                
        cv2.imshow('img',mask)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

if __name__ == "__main__":
    main()
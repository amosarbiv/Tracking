import numpy as np
import time
import cv2
from utils.mouse import Selector
import os


class CrossCor():

    def acquireTarget(self, picture, x, y, w, h):
        target = np.copy(picture[y:y+h, x:x+w])
        return target

    def vcorrcoef(self, target, window):
        X = np.copy(target)
        y = np.copy(window)
        #handling some size issues
        X_x, X_y = X.shape
        y_x, y_y = y.shape
        finalXSize, finalYSize = 0, 0
        if (X_x > y_x):
            finalXSize = y_x
        else:
            finalXSize = X_x
        
        if (X_y > y_y):
            finalYSize = y_y
        else:
            finalYSize = X_y

        X.resize((finalXSize, finalYSize))
        y.resize((finalXSize, finalYSize))
        
        X = X.astype(dtype='float64')
        y = y.astype(dtype='float64')
        Xm = X.mean()
        ym = y.mean()
        r_num = np.sum((X-Xm)*(y-ym))
        r_den = np.sqrt(np.sum(np.square(X-Xm))) * \
                        np.sqrt(np.sum(np.square(y-ym)))
        r = r_num/r_den
        return r

    def getSearchPoints(self, x, y, w, h, imgSizes):
        # calculating end and satr x point
        if (x-w < 0):
            startXPoint = 0
        else:
            startXPoint = x-w
        if (x+w > imgSizes[1]):
            endXPoint = imgSizes[1]
        else:
            endXPoint = x+w

        # calculating end and strt y point
        if (y-h < 0):
            startYPoint = 0
        else:
            startYPoint = y-h
        if (y+h > imgSizes[0]):
            endYPoint = imgSizes[0]
        else:
            endYPoint = y+h

        return (startXPoint, endXPoint, startYPoint, endYPoint)

    def Track(self):

        s = "video//00000"
        frame = cv2.imread(os.path.join("video", "00000004.jpg"))
        x, y, w, h, window_name = Selector(frame).accuireTarget()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        target = self.acquireTarget(image, x, y, w, h)
        cv2.destroyAllWindows()
        cnt = 0
        workingX = x
        workingY = y
        for i in range(4, 793):
            # just for reading the picture
            num=str(i)
            if (len(num) < 3):
                num='0'*(3-len(num)) + num
            frame=cv2.imread(s+num+".jpg")

            img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            startXPoint, endXPoint, startYPoint, endYPoint=self.getSearchPoints(
                workingX, workingY, w, h, img_gray.shape)

            # the big calculation section calculating the cross corr coefficient
            #start=time.time()
            maxX, maxY = 0, 0
            maxCrossCoeff=0
            for currentX in range(startXPoint, endXPoint, 10):
                for currentY in range(startYPoint, endYPoint, 10):
                    crossCoeff=self.vcorrcoef(
                        target, img_gray[currentY:currentY+h, currentX:currentX+w])
                    if (crossCoeff > maxCrossCoeff):
                        maxCrossCoeff=crossCoeff
                        maxX=currentX
                        maxY=currentY
            print(maxCrossCoeff)
            #print(time.time() - start)

            # the new target is where we found the maximum cross corr coefficient
            #target=img_gray[maxY:maxY+h, maxX:maxX+w]
            workingX = maxX 
            workingY = maxY

            # dispalying the picture with the rectangle around the target
            cv2.rectangle(frame, (workingX, workingY), (workingX + w, workingY + h), (0, 0, 255), 1)
            cv2.imshow('new', frame)
            cnt += 1
            time.sleep(1)
            k=cv2.waitKey(1) & 0xFF
            if k == 27:
                break

if __name__ == "__main__":
    tracker=CrossCor()
    tracker.Track()

import numpy as np
import cv2
from collections import defaultdict

class CrossCorTracker():
    def __init__(self):
        self.windowPixelJump = 4
        self.coeffDict = defaultdict()

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

    def Track(self, img, target,startXPoint, endXPoint, startYPoint, endYPoint):
        self.coeffDict = defaultdict()
        h = target.shape[0]
        w = target.shape[1]
        # the big calculation section calculating the cross corr coefficient
        #start=time.time()
        maxX, maxY = 0, 0
        maxCrossCoeff=0
        cv2.imshow('in corr', img)
        cv2.waitKey(1)
        for currentX in range(startXPoint, endXPoint, self.windowPixelJump):
            for currentY in range(startYPoint, endYPoint, self.windowPixelJump):
                if ((currentX + w) > endXPoint):
                    currentX = endXPoint - w
                if ((currentY + h) > endYPoint):
                    currentY = endYPoint - h
                # cv2.rectangle(img, (currentX, currentY), (currentX+w, currentY+h), (255,255,255), 1)
                crossCoeff=abs( self.vcorrcoef(
                    target, img[currentY:currentY+h, currentX:currentX+w]))
                self.coeffDict[(currentX, currentY)] = crossCoeff
                if (crossCoeff > maxCrossCoeff):
                    maxCrossCoeff=crossCoeff
                    maxX=currentX
                    maxY=currentY
        #print(maxCrossCoeff)
        return (maxCrossCoeff, maxX, maxY)
        #print(time.time() - start)
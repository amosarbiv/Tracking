import numpy as np
from scipy import signal
import time
import cv2
from scipy import fftpack
from utils.mouse import Selector
from skimage.feature import match_template



class CrossCor():

    def acquireTarget(self,picture,x,y,w,h):
        target = np.copy(picture[x:x+w,y:y+h])
        return target
    
    def vcorrcoef(self, X,y):
        Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
        ym = np.mean(y)
        r_num = np.sum((X-Xm)*(y-ym),axis=1)
        r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
        r = r_num/r_den
        return r

    def Track(self):
        
        s = "video//00000"
        frame = cv2.imread("video//00000004.jpg")
        x, y, w, h, window_name = Selector(frame).accuireTarget()
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        target = self.acquireTarget(image, x,y,w,h)
        cv2.destroyAllWindows()
        cnt = 0
        
        for i in range(4,793):
            #just for reading the picture
            num = str(i)
            if (len(num) < 3):
                    num = '0'*(3-len(num)) + num
            frame = cv2.imread(s+num+".jpg") 

            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            start = time.time()
            result = self.cor(img_gray, target)
            print(x,y)
            print(time.time() - start)
            ij = np.unravel_index(np.argmax(result), result.shape)
            x, y = ij[::-1]
            w,h = target.shape[::-1]
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 1)
            cv2.imshow('new',frame)
            cnt+=1
            prev_frame = img_gray
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

if __name__ == "__main__":
    tracker = CrossCor()
    tracker.Track()
import cv2 
import numpy as np

class fastMeanTracker():
    def __init__(self, firstCroped):
        self.term_crit = ( cv2.TERM_CRITERIA_EPS, 1, 1 )
        hsv_roi = cv2.cvtColor(firstCroped, cv2.COLOR_BGR2HSV)  
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180]) 
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)


    def track(self, ROI, croped):
        
        hsv = cv2.cvtColor(croped, cv2.COLOR_BGR2HSV)
        
        dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, ROI, self.term_crit)
        print(ret)
        x,y,w,h = track_window
        cv2.imshow('in', croped[y:y+h,x:x+w])
        cv2.waitKey()
        cv2.destroyWindow('in')
        return x,y,w,h

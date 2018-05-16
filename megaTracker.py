import cv2
import numpy as np
import logging
from collections import defaultdict
import argparse

#local imports
import CrossCorTracker # should be somewhere else
from utils.trackable import Trackable
from utils.mouse import Selector
from utils.utils import readImages

#trying to correct the kalman filter
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class megaTracker():
    def __init__(self, videoPath, delay):
        #environment inits
        self.logger = logging.getLogger("megaTracker")
        self.delay = delay
        #self.videoPath = videoPath
        self.frames = readImages(videoPath)
        self.finalWindowName = "FinalShow"

        #tracking inits
        self.maxCoeff = 0
        self.targets = defaultdict()
        self.crossCorTracker = CrossCorTracker.CrossCorTracker()

        self.userROISelection()
        cv2.destroyAllWindows()

    def showImages(self, frame, boundingBoxes):
        colors = [(0, 0, 255), (255, 0, 255), (255, 255, 255)]
        
        cnt = 0
        for box in boundingBoxes:
            color = colors[cnt]
            cv2.rectangle(frame, box.top_left(),
                        box.bottom_right(), color, 1)
            cnt += 1
        
        cv2.imshow(self.finalWindowName, frame)
        cv2.waitKey(self.delay)

    def isOccluded(self, currentCoeff):
        ratio = currentCoeff / self.maxCoeff 
        self.logger.debug("isOccluded ratio: %f" % ratio)
        if ( ratio < 0.7 ):
            return True
        else:
            return False

    def initKalman(self, center, dt=1):
        #new kalman filter
        my_filter = KalmanFilter(dim_x=6, dim_z=2)
        
        my_filter.x =  center   # initial state (location and velocity)

        my_filter.F = np.array([[1, dt, .5*dt*dt,0,0,0],
                                        [0,1,dt,0,0,0],
                                        [0,0,1,0,0,0],
                                        [0,0,0,1, dt, .5*dt*dt],
                                        [0,0,0,0,1,dt],
                                        [0,0,0,0,0,1]])    # state transition matrix
        H = np.zeros((2,6))
        H[0,0] = H[1,3] = 1
        my_filter.H = H    # Measurement function
        my_filter.P *= 10.                 # covariance matrix
        my_filter.R = 5                      # state uncertainty
        my_filter.Q = Q_discrete_white_noise(2, dt=1, var=0.5, block_size=3) # process uncertainty
        
        self.kalman = my_filter

    def cropFromTrackable(self, frame, trackObj):
        startX = trackObj.top_left()[0]
        startY = trackObj.top_left()[1]
        finishX = trackObj.bottom_right()[0]
        finishY = trackObj.bottom_right()[1]
        res = frame[startY:finishY, startX:finishX]
        return res

    def userROISelection(self):
        #selecting the first Target
        frame = next(self.frames)[0]
        x,y,w,h, window_name = Selector(frame).accuireTarget()
        firstTarget = Trackable(box=(x, y, w, h))      
        center = np.zeros((6, 1))
        center[0,0] = firstTarget.center[0]
        center[3,0] = firstTarget.center[1]
        self.initKalman(center)
        
        self.lastTracking = firstTarget

        self.targets['firstTarget'] = self.toGray(self.cropFromTrackable(frame, firstTarget))
        self.firstTargetWidth = w
        self.firstTargetheight = h

    def toGray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def corrCoeff(self, frame, trackingWindow):
        localMax = 0
        localX = 0
        localY = 0
        for targetLabel,target in self.targets.items():
            coeff, x, y = self.crossCorTracker.Track(
                        frame, target, trackingWindow.top_left()[0], trackingWindow.bottom_right()[0], trackingWindow.top_left()[1], trackingWindow.bottom_right()[1])
            if (coeff > localMax):
                localMax = coeff
                localX = x
                localY = y
        return localMax, localX, localY

    def kalmanStep(self,occluded, currentTracking):
        #we need to decide if kalman is our last tracking or our measurement
        self.kalman.update(currentTracking.center.reshape(2,1))
        self.kalman.predict()
        prediction = np.dot(self.kalman.H, self.kalman.x).flatten().astype(np.int)
        if (occluded):
            self.lastTracking = Trackable(center=prediction)
        else:
            self.lastTracking = currentTracking


    def mainLoop(self):
        for frame, *_ in self.frames:
            grayFrame = self.toGray(frame)
            crop, trackingWindow = self.lastTracking.tracking_window(grayFrame)

            #getting the correlation coeff and upper left dot
            currentCoeff, x, y = self.corrCoeff(grayFrame, trackingWindow)

            self.maxCoeff = max(self.maxCoeff, currentCoeff)
            
            occluded = self.isOccluded(currentCoeff)

            currentTracking = Trackable(box=(x,y,self.firstTargetWidth,self.firstTargetheight))
            
            self.kalmanStep(occluded, currentTracking)

            boundingBoxes = list()
            boundingBoxes.append(self.lastTracking) 
            self.showImages(frame, boundingBoxes)

def main():
    tracker = megaTracker(args.videoPath, args.delay) 
    tracker.mainLoop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--videoPath')
    parser.add_argument('-d', '--delay', type=int, default=1)
    args = parser.parse_args()
    main()




    
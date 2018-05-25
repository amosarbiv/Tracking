import cv2
import numpy as np
import logging
from collections import defaultdict
import argparse
import torch

#local imports
import CrossCorTracker # should be somewhere else
from utils.trackable import Trackable
from utils.mouse import Selector
from utils.utils import readImages
import fastMeanTracker
from detect import load_yolo, track

#trying to correct the kalman filter
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class megaTracker():
    def __init__(self, videoPath, delay, cfgfile, weightfile, cuda):
        #environment inits
        self.logger = logging.getLogger("megaTracker")
        self.delay = delay
        #self.videoPath = videoPath
        self.frames = readImages(videoPath)
        self.finalWindowName = "FinalShow"

        #tracking inits
        self.maxCoeff = 0
        self.lamda = 0.5
        self.targets = defaultdict()
        self.crossCorTracker = CrossCorTracker.CrossCorTracker()
        self.darknet = load_yolo(cfgfile, weightfile, cuda)

        self.userROISelection()
        cv2.destroyAllWindows()

    def showImages(self, frame, boundingBoxes):
        colors = [(255, 0, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 255, 255)]

        cnt = 0
        for box in boundingBoxes:
            color = colors[cnt]
            cv2.rectangle(frame, box.top_left(),
                        box.bottom_right(), color, 1)
            cnt += 1

        cv2.imshow(self.finalWindowName, frame)
        cv2.waitKey(self.delay)

    def isRubbish(self, currentCoeff):
        ratio = currentCoeff / self.maxCoeff
        self.logger.debug("isRubbish ratio: %f" % ratio)
        if ( ratio < 0.6 ):
            return True
        else:
            return False

    def initKalman(self, center, dt=1):
        #new kalman filter
        kalman = cv2.KalmanFilter(4,2)
        kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
        self.kalman = kalman
# my_filter = KalmanFilter(dim_x=6, dim_z=2)
        #
        # my_filter.x =  center   # initial state (location and velocity)
        #
        # my_filter.F = np.array([[1, dt, .5*dt*dt,0,0,0],
        #                                 [0,1,dt,0,0,0],
        #                                 [0,0,1,0,0,0],
        #                                 [0,0,0,1, dt, .5*dt*dt],
        #                                 [0,0,0,0,1,dt],
        #                                 [0,0,0,0,0,1]])    # state transition matrix
        # H = np.zeros((2,6))
        # H[0,0] = H[1,3] = 1
        # my_filter.H = H    # Measurement function
        # my_filter.P *= 10.                 # covariance matrix
        # my_filter.R = 5                      # state uncertainty
        # my_filter.Q = Q_discrete_white_noise(2, dt=1, var=0.5, block_size=3) # process uncertainty
        #
        # self.kalman = my_filter

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

        self.fastMeanTracker = fastMeanTracker.fastMeanTracker(self.cropFromTrackable(frame,firstTarget))

        self.lastTracking = firstTarget

        self.targets['firstTarget'] = self.toGray(self.cropFromTrackable(frame, firstTarget))
        self.firstTargetWidth = w
        self.firstTargetheight = h

    def toGray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def diffrentFromFirst(self):
        coeff = self.crossCorTracker.vcorrcoef(self.targets['firstTarget'], self.targets['lastTarget'])
        coeff = abs(coeff)
        self.logger.debug('diffrent coeff: %f' % coeff)
        if (coeff < 0.4):
            del self.targets['lastTarget']
            self.lamda = 0.3
            self.logger.debug('removed last target')
        else:
            self.lamda = 0.65


    def corrCoeff(self, frame, trackingWindow):
        coeffs = {}
        localMax = 0
        localX = 0
        localY = 0
        label = ""
        for targetLabel, target in self.targets.items():
            coeff, x, y = self.crossCorTracker.Track(
                        frame, target, trackingWindow.top_left()[0], trackingWindow.bottom_right()[0], trackingWindow.top_left()[1], trackingWindow.bottom_right()[1])
            # coeffs[targetLabel] = self.crossCorTracker.coeffDict
            if (coeff > localMax):
                localMax = coeff
                localX = x
                localY = y
                label = targetLabel

        # for (x,y), coeff_1 in coeffs['firstTarget'].items():
        #     coeff = (1-self.lamda)*coeff_1 + self.lamda*coeffs['lastTarget'][(x,y)]

        self.logger.debug('label won: %s' % label)
        return localMax, localX, localY

    def kalmanStep(self, visible, currentTracking, grayFrame):
        #we need to decide if kalman is our last tracking or our measurement
        tp = self.kalman.predict()
        # prediction = np.dot(self.kalman.H, self.kalman.x).flatten().astype(np.int)
        prediction = Trackable(center=np.array([int(tp[0]),int(tp[1])]))
        self.previousTracking = self.lastTracking
        if visible:
            self.lastTracking = currentTracking
            self.targets['lastTarget'] = self.cropFromTrackable(grayFrame, currentTracking)
        else:
            self.lastTracking = prediction

        # self.kalman.update(self.lastTracking.center.reshape(2,1))
        self.kalman.correct(self.lastTracking.center.reshape(2,1).astype(np.float32))
        return True

    def isOccluded(self, kalmanPrediction):
        kalmanCoeff = 0
        for targetLabel,target in self.targets.items():
            coeff = self.crossCorTracker.vcorrcoef(target, kalmanPrediction)
            if (coeff > kalmanCoeff):
                kalmanCoeff = coeff
        
        kalmanCoeff = abs(kalmanCoeff)
        if (kalmanCoeff < 0.7):
            return True
        else:
            return False

    def mainLoop(self):
        ocludded_duration = 0
        normalSearchWindow = True
        for frame, *_ in self.frames:
            grayFrame = self.toGray(frame)
            if normalSearchWindow:
                crop, trackingWindow = self.lastTracking.tracking_window(grayFrame)
            else:
                crop, trackingWindow = self.lastTracking.tracking_window(grayFrame, scale=2)
            boxes = track(self.darknet, frame)
            trackables = Trackable.from_yolo(boxes, frame, trackingWindow)

            max_coeff, max_ind = -1, -1
            for i, box in enumerate(trackables):
                prediction_frame = self.toGray(self.cropFromTrackable(frame, box))
                for targetLabel, target in self.targets.items():
                    coeff = self.crossCorTracker.vcorrcoef(target, prediction_frame)

                    if (coeff > max_coeff):
                        max_coeff = coeff
                        max_ind = i

            self.logger.debug("max_coeff: %f" % max_coeff)
            if max_coeff < 0.2:
                ocludded_duration += 1
            else:
                ocludded_duration = 0

            visible = (ocludded_duration == 0) and max_ind != -1
            best_track = trackables[max_ind] if visible else None
            normalSearchWindow = self.kalmanStep(visible, best_track, grayFrame)
            self.logger.debug("normal search window: %r" % normalSearchWindow)

            boundingBoxes = list()
            boundingBoxes.append(self.lastTracking)
            boundingBoxes.append(trackingWindow)
            self.showImages(frame, boundingBoxes)
            self.logger.debug('############')
def main():
    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO), format='[%(asctime)s] %(name)-12s: %(levelname)-8s %(message)s')
    tracker = megaTracker(args.videoPath, args.delay, args.config, args.model, False)
    tracker.mainLoop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--videoPath', required=True)
    parser.add_argument('-d', '--delay', type=int, default=1)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args = parser.parse_args()
    main()




    
import cv2
import numpy as np
import copy

from utils.backgroungRemove import BackGroundSubtractor, denoise
from utils.mouse import Selector
from utils.utils import readImages, save_obj, bb_intersection_over_union
from utils.trackable import Trackable
from trackers.canny import bounding_boxes
from estimators.kalman import kalmanFilter
import CrossCorTracker

#trying to correct the kalman filter
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from scipy.stats import multivariate_normal


def prob_from_kalman(corr_locs, kalman):
    for (x,y), v in corr_locs.items():
        prob = multivariate_normal.pdf(np.array([x,y]), mean=kalman.x_prior, cov=kalman.P_prior)
        corr_locs[(x,y)] = (v, prob)


def showImage(frame, boundingBoxes, delay=1):
    window_name = "Final Show"
    colors = [(0, 0, 255), (255, 0, 255), (255, 255, 255)]
    cnt = 0

    for box in boundingBoxes:
        color = colors[cnt]
        cv2.rectangle(frame, box.top_left(),
                      box.bottom_right(), color, 1)
        cnt += 1
    
    cv2.imshow(window_name, frame)
    cv2.waitKey(delay)

def isOccluded(maxCoeff, currentCoeff):
    precent = currentCoeff / maxCoeff 
    print("precent: %f" % precent)
    if ( precent < 0.7 ):
        return True
    else:
        return False

def initKalman(center, dt=1):
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
    return my_filter

def main():

    frames = readImages(args.images_path)
    frame = next(frames)[0]
    back_subtractor = BackGroundSubtractor(frame)
    x,y,w,h, window_name = Selector(frame).accuireTarget()
    #y, x, h, w, window_name = 208, 247, 105, 60, 'Image'
    corrections = list()
    predictions = list()
    track = list()

    #initialize the needed trackers
    crossCorTracker = CrossCorTracker.CrossCorTracker()
    maxCoeff = 0 

    track.append(Trackable(box=(x, y, w, h)))
    center = np.zeros((6, 1))
    center[0,0] = track[0].center[0]
    center[3,0] = track[0].center[1]
    kalmanPre = initKalman(center)
    kalman = kalmanFilter(center)
    corrected = track[-1]
    kalman.correct(corrected.center)
    kalman = kalman.predict()
    #TODO Amos's Hacks....need to be removed!!!!
    new_meas = False
    target = None
    for frame, *_ in frames:
        new_meas = False
        last_tracking = track[-1]
        mask = back_subtractor.get_binary(frame)
        if target is None:
            target = np.copy(frame[y:y+h, x:x+w])
            target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            target = denoise(target)
            cv2.imshow('target', target)
        new_tracking = False
        while not new_tracking:
            if args.method == 'correlation':                
                #getting the tracking window from the lest measurment
                crop, tracking_window = last_tracking.tracking_window(frame)
                
                #a print to see the tracking window in the real img
                #cv2.rectangle(frame, tracking_window.top_left(), tracking_window.bottom_right(), (255, 255,255), 1)
                #cv2.imshow('frame', frame)
                #cv2.waitKey()
                
                img_gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_gray = denoise(img_gray)
                # img_gray= denoise(mask)   
                
                #getting the cross correlation coefficient and top left corner
                coeff, x, y = crossCorTracker.Track(
                    img_gray, target, tracking_window.top_left()[0], tracking_window.bottom_right()[0], tracking_window.top_left()[1], tracking_window.bottom_right()[1])
                print("in tracker coeff is: %f" %coeff)
                maxCoeff = max(coeff, maxCoeff)
                occluded = isOccluded(maxCoeff, coeff)
                #last tracking and new tracking becomes where we've seen the target
                corr_tracking = Trackable(box=(x,y,w,h))
                new_tracking  = corr_tracking
                # crop, tracking_window = corrected.tracking_window(mask)
                # new_meas = tracking_window.center_of_mass(crop)
                # cv2.imshow('crop', crop)


        prediction = kalmanFilter.to_trackable(kalman.prior)
        predictions.append(prediction)
        """
        if not new_tracking or occluded:
            new_tracking = prediction
        """
        kalmanPre.predict()
        prob_from_kalman(crossCorTracker.coeffDict, kalmanPre)
        print(crossCorTracker.coeffDict)
        kalmanPre.update(new_tracking.center.reshape(2,1))
        prediction = np.dot(kalmanPre.H, kalmanPre.x).flatten().astype(np.int)
        if ( occluded ):
            print("is occluded")
            new_tracking = Trackable(center=prediction) 
        # diagonal = 2 * \
        #     np.linalg.norm(
        #         np.array(track[-1].top_left())-np.array(track[-1].bottom_right()))
        
        # if new_meas.distance(new_tracking) > diagonal:
        #     new_meas = new_tracking
        
        corrected, error = kalman.correct(new_tracking.center)
        kalman = kalman.predict()
        predictedTarget = img_gray[corrected.top_left()[1]:corrected.top_left()[1]+h,corrected.top_left()[0]:corrected.top_left()[0]+w]
        corp , tracking_window = corrected.tracking_window(frame)

        """
        corectedCoeff, x, y = crossCorTracker.Track(
                    img_gray, target, tracking_window.top_left()[0], tracking_window.bottom_right()[0], tracking_window.top_left()[1], tracking_window.bottom_right()[1])
        
        corectedCoeff, x, y = crossCorTracker.Track(
                    img_gray, target, corrected.top_left()[0]-10, corrected.bottom_right()[1]+10, corrected.top_left()[1]-10, corrected.bottom_right()[1]+10)
        
        if (corectedCoeff > coeff):
            print("corrected won!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            last_tracking =  Trackable(center=corrected.center)
            new_tracking =  Trackable(center=corrected.center)
        """
        
        # if (isOccluded(maxCoeff, coeff)):
        #     new_tracking = Trackable(center=corrected.center)
        
        track.append(new_tracking)
        corrections.append(corrected)
        
        #viewing the frames
        boundingBoxes = list()
        boundingBoxes.append(new_tracking)
        boundingBoxes.append(tracking_window)
        boundingBoxes.append(corr_tracking)
        showImage(frame, boundingBoxes, delay=args.delay)
 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images-path')
    parser.add_argument('-d', '--delay', type=int, default=1)
    parser.add_argument('-s', '--frame-size', type=int, default=200)
    parser.add_argument('-m', '--method', type=str,
                        choices=['mass', 'correlation', 'edges'], default='mass')
    args = parser.parse_args()
    main()

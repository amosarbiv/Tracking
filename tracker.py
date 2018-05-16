import cv2
import numpy as np
import copy

from utils.backgroungRemove import BackGroundSubtractor, denoise
from utils.mouse import Selector
from utils.utils import readImages, save_obj, bb_intersection_over_union
from utils.trackable import Trackable
from trackers.canny import bounding_boxes
from estimators.kalman import KalmanFilter
import CrossCorTracker

def showImage(frame, boundingBoxes):
    window_name = "Final Show"
    colors = [(0, 0, 255), (255, 0, 255), (255, 255, 255)]
    cnt = 0

    for box in boundingBoxes:
        color = colors[cnt]
        cv2.rectangle(frame, box.top_left(),
                      box.bottom_right(), color, 1)
        cnt += 1
    
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)

def isOccluded(maxCoeff, currentCoeff):
    precent = currentCoeff / maxCoeff 
    print("precent: %f" % precent)
    if ( precent < 0.7 ):
        return True
    else:
        return False

def main():
    frames = readImages(args.images_path)
    frame = next(frames)[0]
    back_subtractor = BackGroundSubtractor(frame)
    x,y,w,h, window_name = Selector(frame).accuireTarget()
    #y, x, h, w, window_name = 208, 247, 105, 60, 'Image'
    corrections = list()
    predictions = list()
    track = list()
    

    maxCoeff = 0 

    track.append(Trackable(box=(x, y, w, h)))
    center = np.zeros((6, 1))
    center[0,0] = track[0].center[0]
    center[3,0] = track[0].center[1]
    kalman = KalmanFilter(center)
    corrected = track[-1]
    kalman.correct(corrected.center)
    kalman = kalman.predict()
    #TODO Amos's Hacks....need to be removed!!!!
    new_meas = False
    target = None
    for frame, *_ in frames:
        last_tracking = track[-1]
        mask = back_subtractor.get_binary(frame)
        if target is None:
            target = np.copy(frame[y:y+h, x:x+w])
            target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            cv2.imshow('target', target)
        new_tracking = False
        while not new_tracking:
            if args.method == 'correlation':
                #initialize the needed trackers
                crossCorTracker = CrossCorTracker.CrossCorTracker()
                
                #getting the tracking window from the lest measurment
                crop, tracking_window = last_tracking.tracking_window(frame)
                
                #a print to see the tracking window in the real img
                #cv2.rectangle(frame, tracking_window.top_left(), tracking_window.bottom_right(), (255, 255,255), 1)
                #cv2.imshow('frame', frame)
                #cv2.waitKey()
                
                img_gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#denoise(mask)   
                
                #getting the cross correlation coefficient and top left corner
                coeff, x, y = crossCorTracker.Track(
                    img_gray, target, tracking_window.top_left()[0], tracking_window.bottom_right()[0], tracking_window.top_left()[1], tracking_window.bottom_right()[1])
                print("in tracker coeff is: %f" %coeff)
                maxCoeff = max(coeff, maxCoeff)
                occluded = isOccluded(maxCoeff, coeff)
                #last tracking and new tracking becomes where we've seen the target
                new_meas = Trackable(box=(x,y,w,h))
                crop, tracking_window = corrected.tracking_window(mask)
                
                #new_meas = tracking_window.center_of_mass(crop)
            new_tracking = True


        prediction = KalmanFilter.to_trackable(kalman.prior)
        predictions.append(prediction)
        if (occluded):
            new_meas = prediction
           

        # diagonal = 2 * \
        #     np.linalg.norm(
        #         np.array(track[-1].top_left())-np.array(track[-1].bottom_right()))
        
        # if new_meas.distance(new_tracking) > diagonal:
        #     new_meas = new_tracking
        
        corrected, error = kalman.correct(new_meas.center)
        kalman = kalman.predict()
        predictedTarget = img_gray[corrected.top_left()[1]:corrected.top_left()[1]+h,corrected.top_left()[0]:corrected.top_left()[0]+w]
        corp , tracking_window = corrected.tracking_window(frame)

        new_tracking = Trackable(center=corrected.center)
        
        track.append(new_tracking)
        corrections.append(corrected)
        
        #viewing the frames
        boundingBoxes = [new_tracking, tracking_window]
        showImage(frame, boundingBoxes)
 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images-path')
    parser.add_argument('-s', '--frame-size', type=int, default=200)
    parser.add_argument('-m', '--method', type=str,
                        choices=['mass', 'correlation', 'edges'], default='mass')
    args = parser.parse_args()
    main()

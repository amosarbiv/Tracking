import cv2

import numpy as np

import CrossCorTracker
from estimators.kalman import KalmanFilter
from goturn.helper.BoundingBox import BoundingBox
from goturn.logger.logger import setup_logger
from goturn.network.regressor import regressor
from goturn.tracker.tracker import tracker
from utils.backgroungRemove import BackGroundSubtractor
from utils.mouse import Selector
from utils.trackable import Trackable
from utils.utils import readImages


def main():
    logger = setup_logger(logfile=None)

    ### init video
    frames = readImages(args.images_path)
    frame = next(frames)[0]

    ### init background subtractor
    back_subtractor = BackGroundSubtractor(frame)

    ### init roi
    x,y,w,h, window_name = Selector(frame).accuireTarget()
    #y, x, h, w, window_name = 208, 247, 105, 60, 'Image'

    ### init trackings and predictions
    corrections = list()
    predictions = list()
    track = list()
    track.append(Trackable(box=(x, y, w, h)))


    ### init kalman
    center = np.zeros((6, 1))
    center[0,0] = track[0].center[0]
    center[3,0] = track[0].center[1]
    kalman = KalmanFilter(center)
    corrected = track[-1]
    kalman.correct(corrected.center)
    kalman = kalman.predict()

    ### init net
    if args.method == 'net':
        objRegressor = regressor('nets/tracker.prototxt', 'nets/models/pretrained_model/tracker.caffemodel',
                                 None, 1, False, logger)
        objTracker = tracker(False, logger) # Currently no idea why this class is needed, eventually we shall figure it out
        # videos = objLoaderVot.get_videos()
        bbox_0 = BoundingBox(x, y, x+w, y+h)
        bbox_0.frame_num = 0
        objTracker.init(frame, bbox_0, objRegressor)



#TODO Amos's Hacks....need to be removed!!!!
    new_meas = False
    target = None
    for i, frame in enumerate(frames):
        frame = frame[0]
        last_tracking = track[-1]
        mask = back_subtractor.get_binary(frame)
        if target is None:
            target = np.copy(frame[y:y+h, x:x+w])
            target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('target', target)
        new_tracking = False
        while not new_tracking:
            if args.method == 'correlation':
                #initialize the needed trackers
                crossCorTracker = CrossCorTracker.CrossCorTracker()
                #getting the tracking window from the lest measurment
                crop, tracking_window = last_tracking.tracking_window(frame)
                #a print to see the tracking window in the real img
                cv2.rectangle(frame, tracking_window.top_left(), tracking_window.bottom_right(), (255, 255,255), 1)

                img_gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#denoise(mask)
                #getting the mass of the current img
                mass = np.sum(img_gray[y:y+h, x:x+w])
                print("mass:", mass)

                #getting the cross correlation coefficient and top left corner
                coeff, x, y = crossCorTracker.Track(
                    img_gray, target, tracking_window.top_left()[0], tracking_window.bottom_right()[0], tracking_window.top_left()[1], tracking_window.bottom_right()[1])
                print("in tracker coeff is: %f" %coeff)
                tragetToNewTarget = crossCorTracker.vcorrcoef(target, img_gray[y:y+h, x:x+w])
                print("tragetToNewTarget: %f" % tragetToNewTarget)
                # if ( tragetToNewTarget < 0.2):
                #     target = img_gray[y:y+h, x:x+w]
                cv2.imshow('target', target)
                #last tracking and new tracking becomes where we've seen the target
                last_tracking = tracking_window
                new_tracking = Trackable(box=(x,y,w,h))
                crop, tracking_window = corrected.tracking_window(mask)
                new_meas = tracking_window.center_of_mass(crop)

            elif args.method == "net":
                bbox = objTracker.track(frame, objRegressor)
                bbox.frame_num = i
                x2, y2, x1, y1 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                h, w =  x2 - x1, y2 - y1
                new_tracking = Trackable(box=(x1, y1, w, h))
                #new_meas = x1 + w/2


            else:  # center of mass
                crop, tracking_window = corrected.tracking_window(mask)
                new_meas = tracking_window.center_of_mass(crop)

                crop, tracking_window = last_tracking.tracking_window(mask)
                new_tracking = tracking_window.center_of_mass(crop)
                last_tracking = tracking_window

        prediction = KalmanFilter.to_trackable(kalman.prior)
        predictions.append(prediction)
        if not new_meas:
            new_meas = prediction
        diagonal = 2 * \
            np.linalg.norm(
                np.array(track[-1].top_left())-np.array(track[-1].bottom_right()))
        if new_meas.distance(new_tracking) > diagonal:
            new_meas = new_tracking
        corrected, error = kalman.correct(new_meas.center)
        kalman = kalman.predict()

        corp , tracking_window = corrected.tracking_window(frame)
        # corectedCoeff, x, y = crossCorTracker.Track(
        #             img_gray, target, tracking_window.top_left()[0], tracking_window.bottom_right()[0], tracking_window.top_left()[1], tracking_window.bottom_right()[1])
        # if (corectedCoeff > coeff):
        #     print("corrected won!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     last_tracking =  Trackable(box=(x,y,w,h))
        #     new_tracking =  Trackable(box=(x,y,w,h))
        track.append(new_tracking)
        corrections.append(corrected)
        cv2.rectangle(frame, new_tracking.top_left(),
                      new_tracking.bottom_right(), (0, 0, 255), 1) # tracker red
        cv2.rectangle(frame, corrected.top_left(),
                   corrected.bottom_right(), (255, 0, 255), 1) #corrected prediction purple
        cv2.rectangle(frame, prediction.top_left(),
                      prediction.bottom_right(), (255, 255, 255), 1) #white box kalman prediction
        cv2.imshow(window_name, frame)
        k=cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    #measurements = [item.as_dict() for item in track]
    #save_obj(measurements, 'measurements')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images-path')
    parser.add_argument('-s', '--frame-size', type=int, default=200)
    parser.add_argument('-m', '--method', type=str,
                        choices=['net', 'mass', 'correlation', 'edges'], default='mass')
    args = parser.parse_args()
    main()

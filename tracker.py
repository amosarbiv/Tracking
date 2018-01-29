import cv2
import numpy as np

from utils.backgroungRemove import BackGroundSubtractor, denoise
from utils.mouse import Selector
from utils.utils import readImages, save_obj, bb_intersection_over_union
from utils.trackable import Trackable
from trackers.canny import bounding_boxes
from estimators.kalman import KalmanFilter

def main():
    frames = readImages(args.images_path)
    frame = next(frames)[0]
    back_subtractor = BackGroundSubtractor(frame)
    # y, x, h, w, window_name = Selector(frame).accuireTarget()
    y, x, h, w, window_name = 208, 247, 105, 60, 'Image'
    corrections = list()
    predictions = list()
    track = list()
    track.append(Trackable(box=(x, y, w, h)))
    kalman = KalmanFilter(np.zeros((6,1)))
    corrected = track[-1]
    kalman.correct(corrected.center)
    kalman = kalman.predict()

    for frame, *_ in frames:
        last_tracking = track[-1]
        mask = back_subtractor.get_binary(frame)
        new_tracking = False
        while not new_tracking:
            if args.method == 'correlation':
                print('correlation not implemented yet')
                break

            elif args.method == 'edges':
                mask = back_subtractor.getMask(frame)
                trackings = bounding_boxes(denoise(mask))
                new_tracking = last_tracking.get_closest(trackings)

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
        diagonal = 2 * np.linalg.norm(np.array(track[-1].top_left())-np.array(track[-1].bottom_right()))
        if new_meas.distance(new_tracking) > diagonal:
            new_meas = new_tracking
        corrected, error = kalman.correct(new_meas.center)
        kalman = kalman.predict()
        track.append(new_tracking)
        corrections.append(corrected)
        cv2.rectangle(frame, new_tracking.top_left(), new_tracking.bottom_right(), (0,0,255), 1)
        cv2.rectangle(frame, corrected.top_left(), corrected.bottom_right(), (255,0,255), 1)
        cv2.rectangle(frame, prediction.top_left(), prediction.bottom_right(), (255,255,255), 1)
        cv2.imshow(window_name, frame)
        cv2.waitKey(0)

    measurements = [item.as_dict() for item in track]
    save_obj(measurements, 'measurements')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images-path')
    parser.add_argument('-s', '--frame-size', type=int, default=200)
    parser.add_argument('-m', '--method', type=str, choices=['mass', 'correlation', 'edges'], default='mass')
    args = parser.parse_args()
    main()

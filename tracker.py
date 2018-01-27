import cv2
import math
import numpy as np

from utils.mouse import Selector
from utils.utils import readImages
from backgroungRemove import BackGroundSubtractor, denoise
from CenterOfMass import centerOfMass


def next_window(x, y, w, h):
    x_new = max(0, x - w)
    y_new = max(0, y - h)
    w_new = 3*w
    h_new = 3*h
    return x_new, y_new, w_new, h_new

def bounding_boxes(frame):
    edged = cv2.Canny(frame, 1, 250)
    # cv2.imshow('new', edged)
    # cv2.waitKey(0)
    # find contours in the edge map
    img, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop over the contours
    objects = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # ensure that the approximated contour is "roughly" rectangular
        if len(approx) < 4:
            continue
        # compute the bounding box of the approximated contour and
        # use the bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        # aspectRatio = w / float(h)

        # compute the solidity of the original contour
        area = cv2.contourArea(c)
        hullArea = max(0.1, cv2.contourArea(cv2.convexHull(c)))
        solidity = area / float(hullArea)

        # compute whether or not the width and height, solidity, and
        # aspect ratio of the contour falls within appropriate bounds
        keepDims = w > 25 and h > 25
        keepSolidity = solidity > 0.3
        # keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

        # ensure that the contour passes all our tests
        if keepDims and keepSolidity:
            # draw an outline around the target and update the status
            # text
            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
            # compute the center of the contour region and draw the
            # crosshairs
            M = cv2.moments(approx)
            (cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            (startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
            (startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
            cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 3)
            cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 3)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 1)
            objects.append(Trackable((x, y, w, h), np.array([cX, cY])))
    # cv2.imshow('Image', frame)
    # cv2.waitKey(0)
    return objects


class Trackable():
    def __init__(self, box, center=None):
        self.x, self.y, self.w, self.h = box
        self.center = self.normalize_center(center) if center is not None\
            else np.array((self.x + self.w / 2, self.y + self.h / 2), dtype=np.uint)

    def box(self):
        return self.x, self.y, self.w, self.h

    def tracking_window(self, frame):
        # tracking window is 4 times in size than last detection
        dims = np.array([self.w, self.h], dtype=np.uint32)
        coords = self.center - dims
        for i in range(len(coords)):
            limit = frame.shape[1-i] - 1 - 2*dims[i]
            coords[i] = int(min(limit, max(0, coords[i])))

        x,y = coords
        w,h = 2*dims
        crop = frame[y:y+h, x:x+w]
        return crop, Trackable(box=(x,y,w,h))

    def normalize_center(self, center):
        return center + np.array([self.x, self.y])

    def distance(self, other):
        return np.linalg.norm(self.center - other.center)

    def get_closest(self, boxes):
        if len(boxes) == 0:
            return self
        best = min(boxes, key=lambda x: x.distance(self))
        return best

    def top_left(self):
        return self.x, self.y

    def bottom_right(self):
        corner = np.array(self.top_left()) + np.array([self.w, self.h])
        return tuple(corner)

def main():
    frames = readImages(args.images_path)
    frame = next(frames)[0]
    back_subtractor = BackGroundSubtractor(frame)
    y, x, h, w, window_name = Selector(frame).accuireTarget()
    # y, x, h, w, window_name = 208, 247, 105, 60, 'Image'
    track = []
    track.append(Trackable(box=(x,y,w,h)))

    for frame, *_ in frames:
        last_tracking = track[-1]
        mask = back_subtractor.getMask(frame)
        # window, window_place = last_tracking.tracking_window(mask)
        # cx, cy = centerOfMass(mask, x, y, h, w)
        boxes = bounding_boxes(denoise(mask))
        new_tracking = last_tracking.get_closest(boxes)
        track.append(new_tracking)
        cv2.rectangle(frame, new_tracking.top_left(), new_tracking.bottom_right(), (0,0,255), 1)
        cv2.imshow(window_name, frame)
        cv2.waitKey(0)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images-path')
    parser.add_argument('-s', '--frame-size', type=int, default=200)
    args = parser.parse_args()
    main()

import numpy as np
from scipy import ndimage
from utils.mouse import Selector

class Trackable():
    def __init__(self, box=None, center=None):
        assert (box is not None or center is not None)
        if box is None:
            w_h = np.array([50, 100])
            x_y = center - w_h/2
            box = np.concatenate([x_y, w_h]).astype(np.int)
        self.x, self.y, self.w, self.h = box
        self.center = center if center is not None \
            else np.array((self.x + self.w / 2, self.y + self.h / 2), dtype=np.int64)

    def box(self):
        return self.x, self.y, self.w, self.h

    def tracking_window(self, frame, scale=1):
        # tracking window is 4 times in size than last detection
        dims = np.array([self.w, self.h], dtype=np.int) * scale
        coords = self.center - dims
        for i in range(len(coords)):
            limit = frame.shape[1-i] - 1 - 2*dims[i]
            coords[i] = int(min(limit, max(0, coords[i])))

        x,y = coords.astype(np.int)
        w,h = (2*dims).astype(np.int)
        crop = frame[y:y+h, x:x+w]
        return crop, Trackable(box=(x,y,w,h))

    def normalize_center(self, center):
        if not isinstance(center, np.ndarray):
            center = np.array(center, dtype=np.int)
        return center + self.top_left()

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

    def center_of_mass(self, frame):
        center = ndimage.measurements.center_of_mass(frame)
        if np.isnan(center).any():
            return False
        # center = tuple(0 if np.isnan(coordinate) else coordinate for coordinate in center)
        center = self.normalize_center(center[::-1]) # numpy images are y,x and we use x,y
        return Trackable(center=center)

    def as_dict(self):
        return {'center': self.center, 'box': self.box()}

    @staticmethod
    def from_yolo(boxes, frame, window):
        result = []
        height, width = frame.shape[:2]
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = (box[0] - box[2]/2.0) * width
            y1 = (box[1] - box[3]/2.0) * height
            x2 = (box[0] + box[2]/2.0) * width
            y2 = (box[1] + box[3]/2.0) * height
            trackable = Trackable((int(x1), int(y1), int(x2-x1), int(y2-y1)))
            if window.inTrackingWindow(trackable):
                result.append(trackable)
        return result

    def inTrackingWindow(self, other):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.x, other.x)
        yA = max(self.y, other.y)
        xB = min(self.x+self.w, other.x+other.w)
        yB = min(self.y+self.h, other.y+other.h)

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # return the intersection over union value
        return interArea > 0

import numpy as np


class Background:

    def __init__(self, frame):
        shape = (*frame.shape, 256)
        self.count = np.zeros(shape, dtype=np.uint8)
        self.update_prob(frame)
        self.image = frame

    def update_prob(self, frame):
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                for c in range(frame.shape[2]):
                    intensity = frame[y,x,c]
                    self.count[y, x, c, intensity] += 1

    def get_image(self):
        return np.argmax(self.count, axis=3).astype(np.uint8)



from deep_learning.train import TrackerNet
from utils.utils import readImages
import torch
import cv2
import numpy as np


class FrameNet:
    def __init__(self, model_path):
        self.model = TrackerNet()
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model.eval()

    def predict(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        factor = image.shape[:2] / np.array([224., 224.])
        img = cv2.resize(img, (224,224))\
                  .transpose(2, 0, 1)\
                  .astype(np.float32) / 255.
        prediction = self.model(img)
        prediction[:, ::2] *= factor[1]
        prediction[:, 1::2] *= factor[0]
        return prediction

def preview(image, prediction):
    pts = prediction.astype(np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image, [pts], True, (0,255,255))
    cv2.imshow(image)

def main():
    model = FrameNet(args.model_path)

    frames = readImages(args.images_path)
    frame = next(frames)[0]
    # y, x, h, w, window_name = Selector(frame).accuireTarget()
    y, x, h, w, window_name = 208, 247, 105, 60, 'Image'
    track = list()
    track.append(Trackable(box=(x, y, w, h)))

    for frame, *_ in frames:


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images-path', required=True)
    parser.add_argument('-p', '--model-path', required=True)
    args = parser.parse_args()
    main()




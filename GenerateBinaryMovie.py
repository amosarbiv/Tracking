import numpy as np
from utils import project_root
import os
import random
from PIL import Image

def firstFrame(w=None, h=None):
    frame = np.zeros((args.frame_size, args.frame_size),dtype=np.uint8)
    w = random.randint(10,int(args.frame_size*0.5)) if w is None else w
    h = random.randint(10,int(args.frame_size*0.5)) if h is None else h
    x = random.randint(0,args.frame_size-w)
    y = random.randint(0,args.frame_size-h)
    target = np.ones((h,w),dtype=np.uint8)*255
    frame[y:y+h, x:x+w] = target
    return frame, x, y, w, h

def sanitizeTarget(x,y,w,h):
    target_x = max(0,min(x,args.frame_size))
    target_y = max(0,min(y,args.frame_size))
    width = w + min(0, x)
    width = min(args.frame_size-(x+1),width)
    hight = h + min(0,y)
    hight = min(args.frame_size-(y+1),hight)
    if hight <=0 or width <=0:
        _, target_x, target_y, width, hight = firstFrame(w, h)
    return target_x, target_y, width, hight


def moveDown(x, y, w, h):
    next_frame = np.zeros((args.frame_size, args.frame_size),dtype=np.uint8)
    target_x , target_y, width, hight = sanitizeTarget(x, y+1, w, h)
    next_frame[target_y:target_y+hight, target_x:target_x+width] = np.ones((hight,width),dtype=np.uint8)*255
    return next_frame, x, y+1

def moveUp(x, y, w, h):
    next_frame = np.zeros((args.frame_size, args.frame_size),dtype=np.uint8)
    target_x , target_y, width, hight = sanitizeTarget(x, y-1, w, h)
    next_frame[target_y:target_y+hight, target_x:target_x+width] = np.ones((hight,width),dtype=np.uint8)*255
    return next_frame, x, y-1

def moveRight(x, y, w, h):
    next_frame = np.zeros((args.frame_size, args.frame_size),dtype=np.uint8)
    target_x , target_y, width, hight = sanitizeTarget(x+1, y, w, h)
    next_frame[target_y:target_y+hight, target_x:target_x+width] = np.ones((hight,width),dtype=np.uint8)*255
    return next_frame, x+1, y

def moveLeft(x, y, w, h):
    next_frame = np.zeros((args.frame_size, args.frame_size),dtype=np.uint8)
    target_x , target_y, width, hight = sanitizeTarget(x-1, y, w, h)
    next_frame[target_y:target_y+hight, target_x:target_x+width] = np.ones((hight,width),dtype=np.uint8)*255
    return next_frame, x-1, y

def frameGenerator(steps=10):
    actions = [moveUp, moveDown, moveLeft, moveRight]
    frame, x, y, w, h = firstFrame()
    while True:
        action = random.sample(actions,1)[0]
        for step in range(steps):
            frame, x, y = action(x, y, w, h)
        yield frame, x, y


def main():
    # video = np.empty((args.number_of_frames, args.frame_size, args.frame_size), dtype=np.uint8)
    # video[0], x, y, w, h = firstFrame()
    # img = Image.fromarray(video[0])
    # img.save(os.path.join(args.output_dir, 'binary_{}.jpg'.format(0)),)
    # i = 1
    # for j in range(30):
    #     video[i], x, y = moveRight(x, y, w, h)
    #     img = Image.fromarray(video[i])
    #     img.save(os.path.join(args.output_dir, 'binary_{}.jpg'.format(i)))
    #     i += 1
    # for j in range(30):
    #     video[i], x, y = moveDown(x, y, w, h)
    #     img = Image.fromarray(video[i])
    #     img.save(os.path.join(args.output_dir, 'binary_{}.jpg'.format(i)))
    #     i += 1
    # for j in range(30):
    #     video[i], x, y = moveLeft(x, y, w, h)
    #     img = Image.fromarray(video[i])
    #     img.save(os.path.join(args.output_dir, 'binary_{}.jpg'.format(i)))
    #     i += 1
    frame_gen = frameGenerator()
    for i in range(100):
        frame, x, y = next(frame_gen)
        img = Image.fromarray(frame)
        img.save(os.path.join(args.output_dir, 'binary_{}.jpg'.format(i)))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', help="path to images directory")
    parser.add_argument('-n', '--number-of-frames',type=int, default=100)
    parser.add_argument('-s', '--frame-size',type=int, default=50)
    parser.add_argument('-r', '--randomized', action='store_true')
    args = parser.parse_args()
    if args.output_dir == None:
        args.output_dir = os.path.join(project_root(), 'binary_target')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main()

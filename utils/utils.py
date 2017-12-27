import os
import glob
import cv2

PROJECT_ROOT = {'orrbarkat': '/Users/orrbarkat/repos/tracking',
                'Amos':'/path/to/your_project'} # replace amos with you username

def project_root():
    return PROJECT_ROOT[os.getlogin()]

def readImages(path, gray=False):
    images = sorted(glob.glob(os.path.join(path, '*.jpg')))
    for image_path in images:
        img = cv2.imread(image_path)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        yield img, 0, 0, 0, 0




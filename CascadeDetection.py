import numpy as np
import cv2
import argparse
import os
import copy

DEFAULT_PATH = "D:\\openCV\\opencv\\build\\etc\\haarcascades\\"

def command_line_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="path to images directory", type=str,action="store",nargs=1, default=[""])
    parser.add_argument("-c", "--cascade_path", help="path to haarcascade location", type=str,action="store",nargs=1, default=[DEFAULT_PATH])
    return vars(parser.parse_args())

def main():

    cmd = command_line_handler()
    
    face_cascades = cv2.CascadeClassifier(os.path.join(cmd["cascade_path"][0],"haarcascade_frontalface_alt2.xml" ))
    profile_cascades = cv2.CascadeClassifier(os.path.join(cmd["cascade_path"][0],"haarcascade_profileface.xml" ))
    
    dir_path = cmd["image"][0]
    
    for path in os.listdir(dir_path):
        
        #opening image
        #img = cv2.imread(cmd["image"][0])
        image_path = os.path.join(dir_path, path)
        print(image_path)
        img = cv2.imread(image_path)
        
        #converting to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #detecting faces
        faces = face_cascades.detectMultiScale(gray, 1.3, 5)
        profiles = profile_cascades.detectMultiScale(gray, 1.3, 5)
        
        #printing rectangles
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        for (x,y,w,h) in profiles:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

        cv2.imshow('img',img)
        cv2.waitKey(0)
        continue
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

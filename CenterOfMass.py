import numpy as np
import scipy.misc
import cv2
import time

def createBinaryMovie():
    i = 0
    for x in range(0,800,100):
        for y in range(0,800,100):
            createBinaryImage(i,x,y)
            i+=1
    return

def createBinaryImage(i,x,y):
    img = np.zeros((2000,2000), dtype=np.uint8)
    img[x:x+500, y:y+500] = np.ones((500, 500), dtype=np.uint8)*255
    scipy.misc.imsave('binary_images//hugeLinear//binaryImage_%d.png'%i, img)


def centerOfMass(img, imgX, imgY, h, w):
    
    (X, Y) = imgX+w , imgY+h
    
    m = img / (np.sum(img))

    dx = np.sum(m,1)
    dy = np.sum(m,0)

    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))
 

    return (cx,cy)


def binary_example():

    #createBinaryMovie()
    
    for i in range(64):
        img = scipy.misc.imread("binary_images//hugeLinear//binaryImage_%d.png"%i)
        
        before = time.time()
        myY, myX=centerOfMass(img,0,0,img.shape[0],img.shape[1])
        after = time.time()
        print((after-before))
        
        cv2.circle(img, (int(myX),int(myY)), 2, (0,0,255), -1)
        cv2.imshow('new',img)
  
        k = cv2.waitKey(1) & 0xFF
        if k == 2000:
            break


if __name__ == "__main__":
    binary_example()


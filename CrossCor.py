import numpy as np
from scipy import signal
import time
import cv2
from scipy import fftpack
from utils.mouse import Selector
from skimage.feature import match_template

def _centered(arr, newshape):

    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def fftconvolve(in1, in2):

    #in1 = np.asarray(in1)
    #in2 = np.asarray(in2)


    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)

    shape = s1 + s2 - 1

    fshape = [fftpack.helper.next_fast_len(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])

    sp1 = np.fft.rfftn(in1, fshape)
    sp2 = np.fft.rfftn(in2, fshape)
    ret = (np.fft.irfftn(sp1 * sp2, fshape)[fslice].copy())
 
    return _centered(ret, s1 - s2 + 1)

def _window_sum_2d(image, window_shape):

    window_sum = np.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1]
                - window_sum[:-window_shape[0] - 1])

    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1]
                - window_sum[:, :-window_shape[1] - 1])

    return window_sum


def _window_sum_3d(image, window_shape):
    window_sum = _window_sum_2d(image, window_shape)

    window_sum = np.cumsum(window_sum,axis = 2)
    window_sum = (window_sum[:, :, window_shape[2]:-1]
                - window_sum[:, :, :-window_shape[2] - 1])

    return window_sum


class CrossCor():

    def acquireTarget(self,picture,x,y,w,h):
        target = np.copy(picture[x:x+w,y:y+h])
        return target
    
    

    #@vectorize([()])
    def cor(self,image, template):

        print(image.shape)
        print(template.shape)
        if image.ndim < template.ndim:
            raise ValueError("Dimensionality of template must be less than or "
                            "equal to the dimensionality of image.")
        if np.any(np.less(image.shape, template.shape)):
            raise ValueError("Image must be larger than template.")

        image_shape = image.shape

        image = np.array(image, dtype=np.float64, copy=False)

        pad_width = tuple((width, width) for width in template.shape)

        image = np.pad(image, pad_width=pad_width, mode='constant',
                        constant_values=0)


        # Use special case for 2-D images for much better performance in
        # computation of integral images
        image_window_sum = _window_sum_2d(image, template.shape)
        image_window_sum2 = _window_sum_2d(image ** 2, template.shape)
            


        template_mean = template.mean()
        template_volume = np.prod(template.shape)
        template_ssd = np.sum((template - template_mean) ** 2)
        start = time.time()
        xcorr = fftconvolve(image, template[::-1, ::-1])[1:-1, 1:-1]
        print(time.time() - start)


        numerator = xcorr - image_window_sum * template_mean

        denominator = image_window_sum2
        np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
        np.divide(image_window_sum, template_volume, out=image_window_sum)
        denominator -= image_window_sum
        denominator *= template_ssd
        np.maximum(denominator, 0, out=denominator)  # sqrt of negative number not allowed
        np.sqrt(denominator, out=denominator)

        response = np.zeros_like(xcorr, dtype=np.float64)

        # avoid zero-division
        mask = denominator > np.finfo(np.float64).eps

        response[mask] = numerator[mask] / denominator[mask]

        
        slices = []
        for i in range(template.ndim):
            d0 = template.shape[i] - 1
            d1 = d0 + image_shape[i] - template.shape[i] + 1
            slices.append(slice(d0, d1))
        return response[slices]

    def Track(self,video):
        
        cap = cv2.VideoCapture(video)

        ret, frame = cap.read()
        x, y, w, h, window_name = Selector(frame).accuireTarget()
        cv2.destroyAllWindows()
        #x,y,w,h = 67 ,125, 362, 223
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #cvuint8 = cv2.convertScaleAbs(image)
        
        target = self.acquireTarget(image, x,y,w,h)

        #target -= target.mean().astype('uint8')
        cnt = 0
        while ( cap.isOpened() ):
            ret, frame = cap.read()

            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #cv2.matchTemplate(img_gray,target,res,cv2.TM_CCOEFF_NORMED)
            
            #result = match_template(img_gray[x-2*w:x+2*w,y-2*h:y+2*h], target)
            #shape = max(0,x-w):min(image.shape[0],x+2*w),max(0,y-h):min(y+2*h,image.shape[1])
            if cnt%2 == 0:
                result = self.cor(img_gray[ max(0,x-w):min(image.shape[0],x+2*w),max(0,y-h):min(y+2*h,image.shape
                [1])], target)
            else:
                result = self.cor(img_gray[max(0,y-h):min(y+2*h,image.shape[1]),max(0,x-w):min(image.shape[0],x+2*w)], target)
            ij = np.unravel_index(np.argmax(result), result.shape)
            x, y = ij[::-1]
            w,h = target.shape[::-1]
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 1)
            #target = np.copy(img_gray[x : x+w, y : y+h], )
            
            cv2.imshow('new',frame)
            
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            cnt +=1

if __name__ == "__main__":
    tracker = CrossCor()
    tracker.Track("/Users/orrbarkat/Downloads/Crowd_PETS09/S2/L1/Time_12-34/View_001/test.mp4")
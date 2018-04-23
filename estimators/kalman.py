import numpy as np
from utils.utils import load_obj
from utils.trackable import Trackable

class KalmanFilter:
    Q = np.eye(6)
    R = np.eye(2)
    C = np.zeros((2,6))
    C[0,0] = C[1,3] = 1

    def __init__(self, prior, error=None, transition=None, dt=1):
        if transition is None:
            transition = np.array([[1, dt, .5*dt*dt,0,0,0],
                                     [0,1,dt,0,0,0],
                                     [0,0,1,0,0,0],
                                     [0,0,0,1, dt, .5*dt*dt],
                                     [0,0,0,0,1,dt],
                                     [0,0,0,0,0,1]])
        assert(transition.shape == (6,6))
        self.prior = prior
        self.transition = transition
        self.P = np.eye(6) * 0.01 if error is None else error
        self.estimate = prior
        self.error_estimation = self.P

    def correct(self, measurement):
        c = KalmanFilter.C
        meas = measurement.reshape((2,1))
        amp = np.dot(self.P , np.dot(c.T , np.linalg.inv(np.dot(c , np.dot(self.P , c.T)) + KalmanFilter.R)))
        error_estimation = np.dot((np.eye(6) - np.dot(amp , c)) , self.P)
        estimation = self.prior + np.dot(amp , (meas - np.dot(c , self.prior)))
        self.estimate = estimation
        self.error_estimation = error_estimation
        return self.to_trackable(estimation), error_estimation

    def predict(self):
        prediction = np.dot(self.transition , self.estimate)
        error_prediction = np.dot(np.dot(self.transition , self.error_estimation) , self.transition.T) + KalmanFilter.Q
        return KalmanFilter(prediction, error_prediction)

    @staticmethod
    def to_trackable(state):
        center = np.dot(KalmanFilter.C , state).reshape(-1).astype(np.int)
        return Trackable(center=center)


def main():
    meas = load_obj('measurments')

    pre = KalmanFilter()



if __name__ == '__main__':
    main()
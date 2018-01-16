import numpy as np
import matplotlib.pyplot as plt

def predict(u,P,F,Q):
    u = np.dot(F,u)
    P = np.dot(F,np.dot(P,F.T)) + Q
    return u, P

def correct(u,A,b,P,Q,R):
    C = np.dot(A, np.dot(P,A.T)) + R
    K = np.dot(P, np.dot(A.T, np.linalg.inv(C)))

    u = u + np.dot(K, (b-np.dot(A,u)))
    P = P - np.dot(K, np.dot(C, K.T))

    return u, P

def main():
    dt = 0.1
    A = np.array([[1,0], [0,1]])
    u = np.zeros((2,1))
    
    # random signal
    b = np.array([[u[0,0] + np.random.randn(1)[0]], [u[1,0] + np.random.randn(1)[0]]])

    P = np.diag((0.01, 0.01))
    F = np.array([[1.0, dt], [0.0, 1.0]])

    Q = np.eye(u.shape[0])
    R = np.eye(b.shape[0])

    N = 100

    predictions, corrections, measurments = [] ,  [] , []

    for k in range(0,N):
        u, P = predict(u,P, F, Q)
        predictions.append(u)
        u,p = correct(u, A,b, P,Q,R)
        corrections.append(u)
        measurments.append(b)
        b = np.array([[u[0,0] + np.random.randn(1)[0]], [u[1,0] + np.random.randn(1)[0]]])
    
    print("predicted final estimate: %f" % predictions[-1][0])
    print("corrected final estimate: %f" % corrections[-1][0])
    print("measured state: %f" % measurments[-1][0])
    
    t = np.arange(0, 100)
    fig = plt.figure(figsize=(40,90))

    axes = fig.add_subplot(2,2,1)
    axes.set_title("Simple Kalman")
    axes.plot(t, np.array(predictions)[0:100, 0], 'o', label='predictions')
    axes.plot(t, np.array(corrections)[0:100, 0], 'X', label='corrections')
    axes.plot(t, np.array(measurments)[0:100, 0], '^', label='measurments')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
import numpy as np

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x):
        self.A = A  # State transition matrix
        self.B = B  # Control input matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance matrix
        self.R = R  # Observation noise covariance matrix
        self.P = P  # Estimate error covariance matrix
        self.x = x  # Initial state estimate

    def predict(self, u=None):
        # Predict state estimate based on previous estimate and control input (if given)
        self.x = np.dot(self.A, self.x)
        if u is not None:
            self.x += np.dot(self.B, u)
        # Predict estimate error covariance matrix
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        # Calculate Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Update state estimate and estimate error covariance matrix
        self.x += np.dot(K, z - np.dot(self.H, self.x))
        self.P = np.dot(np.eye(self.P.shape[0]) - np.dot(K, self.H), self.P)
        return self.x

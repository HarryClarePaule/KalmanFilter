import matplotlib.pyplot as plt
import numpy as np
from kalmanfilter import KalmanFilter

# Generate noisy sine wave data
t = np.arange(0, 10, 0.1)
y_true = np.sin(t)
y_noisy = y_true + np.random.normal(0, 0.1, len(t))

# Define Kalman filter parameters
A = np.array([[1, 0.1], [0, 1]])  # State transition matrix
B = None  # Control input matrix
H = np.array([[1, 0]])  # Observation matrix
Q = np.array([[0.01, 0], [0, 0.01]])  # Process noise covariance matrix
R = np.array([[0.1]])  # Observation noise covariance matrix
P = np.eye(2)  # Initial estimate error covariance matrix
x = np.array([[0], [0]])  # Initial state estimate

# Create Kalman filter instance
kf = KalmanFilter(A, B, H, Q, R, P, x)

# Filter noisy sine wave signal
y_filtered = []
for i in range(len(y_noisy)):
    # Predict next state estimate
    kf.predict()
    # Update state estimate based on observation
    kf.update(np.array([[y_noisy[i]]]))
    # Append filtered signal estimate
    y_filtered.append(kf.x[0, 0])

# Plot results
plt.plot(t, y_true, label='True signal')
plt.plot(t, y_noisy, label='Noisy signal')
plt.plot(t, y_filtered, label='Filtered signal')
plt.legend()
plt.show()


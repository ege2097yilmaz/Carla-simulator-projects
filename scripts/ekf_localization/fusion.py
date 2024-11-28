import numpy as np
from data_processing import SensorDataProcessor

class KF:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance

    def predict(self, F, dt, accel, gyro):
        """
        Predict the next state using IMU data.
        
        Parameters:
        - F: State transition matrix.
        - dt: Time step (delta time).
        - accel: Linear acceleration [ax, ay].
        - gyro: Angular velocity (yaw rate) [d_psi].
        """
        # Extract yaw and yaw rate from the state
        psi = self.state[3]  # Current yaw

        self.state[3] = KF.normalize_angle(self.state[3])

        yaw_rate = gyro[2]   # Gyro's z-axis angular velocity
        
        # Predict position and velocity using acceleration and yaw rate
        self.state[0] += self.state[2] * np.cos(psi) * dt  # x += v * cos(psi) * dt
        self.state[1] += self.state[2] * np.sin(psi) * dt  # y += v * sin(psi) * dt
        self.state[2] += accel[0] * dt                     # v += ax * dt
        self.state[3] += yaw_rate * dt                    # psi += yaw_rate * dt
        self.state[4] = yaw_rate                          # Update yaw rate

        # Update the covariance matrix
        self.covariance = np.dot(F, np.dot(self.covariance, F.T)) + self.Q


    def update(self, z, H):
        y = z - np.dot(H, self.state)  # Measurement residual
        S = np.dot(H, np.dot(self.covariance, H.T)) + self.R  # Residual covariance
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))  # Kalman Gain
        self.state = self.state + np.dot(K, y)
        I = np.eye(self.covariance.shape[0])
        self.covariance = np.dot(I - np.dot(K, H), self.covariance)

    @staticmethod
    def normalize_angle(angle):
        """
        Normalize an angle to the range [-pi, pi].

        Parameters:
        - angle: The angle to normalize, in radians.

        Returns:
        - The normalized angle within [-pi, pi].
        """
        normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return normalized_angle
    

class EKF:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance

    def predict(self, dt, accel, gyro):
        """
        Predict the next state using IMU data and non-linear state transition.

        Parameters:
        - dt: Time step (delta time).
        - accel: Linear acceleration [ax, ay].
        - gyro: Angular velocity (yaw rate) [d_psi].
        """
        # Non-linear state transition function
        self.state = self.state_transition(self.state, dt, accel, gyro)

        # Compute Jacobian of the state transition function
        F = self.state_transition_jacobian(self.state, dt, accel, gyro)

        # Update covariance matrix
        self.covariance = np.dot(F, np.dot(self.covariance, F.T)) + self.Q

    def update(self, z, measurement_function, measurement_jacobian):
        """
        Update the state using measurement.

        Parameters:
        - z: Measurement vector.
        - measurement_function: Function to compute predicted measurement from state.
        - measurement_jacobian: Function to compute Jacobian of the measurement function.
        """
        # Compute the predicted measurement
        z_pred = measurement_function(self.state)

        # Compute the measurement Jacobian
        H = measurement_jacobian(self.state)

        # Compute the residual
        y = z - z_pred

        # Compute the residual covariance
        S = np.dot(H, np.dot(self.covariance, H.T)) + self.R

        # Compute the Kalman gain
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))

        # Update the state
        self.state = self.state + np.dot(K, y)

        # Update the covariance matrix
        I = np.eye(self.covariance.shape[0])
        self.covariance = np.dot(I - np.dot(K, H), self.covariance)

    @staticmethod
    def state_transition(state, dt, accel, gyro):
        """
        Non-linear state transition function.

        Parameters:
        - state: Current state [x, y, v, psi, psi_dot].
        - dt: Time step.
        - accel: Linear acceleration [ax, ay].
        - gyro: Angular velocity (yaw rate) [d_psi].

        Returns:
        - Updated state vector.
        """
        x, y, v, psi, _ = state
        yaw_rate = gyro[2]

        # Update state based on dynamics
        x_new = x + v * np.cos(psi) * dt
        y_new = y + v * np.sin(psi) * dt
        v_new = v + accel[0] * dt
        psi_new = psi + yaw_rate * dt
        psi_dot_new = yaw_rate

        # Normalize the yaw angle
        psi_new = EKF.normalize_angle(psi_new)

        return np.array([x_new, y_new, v_new, psi_new, psi_dot_new])

    @staticmethod
    def state_transition_jacobian(state, dt, accel, gyro):
        """
        Jacobian of the state transition function.

        Parameters:
        - state: Current state [x, y, v, psi, psi_dot].
        - dt: Time step.
        - accel: Linear acceleration [ax, ay].
        - gyro: Angular velocity (yaw rate) [d_psi].

        Returns:
        - Jacobian matrix of the state transition.
        """
        _, _, v, psi, _ = state

        # Partial derivatives
        F = np.eye(5)
        F[0, 2] = np.cos(psi) * dt  # ∂x/∂v
        F[0, 3] = -v * np.sin(psi) * dt  # ∂x/∂psi
        F[1, 2] = np.sin(psi) * dt  # ∂y/∂v
        F[1, 3] = v * np.cos(psi) * dt  # ∂y/∂psi
        F[3, 4] = dt  # ∂psi/∂psi_dot

        return F

    @staticmethod
    def normalize_angle(angle):
        """
        Normalize an angle to the range [-pi, pi].

        Parameters:
        - angle: The angle to normalize, in radians.

        Returns:
        - The normalized angle within [-pi, pi].
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
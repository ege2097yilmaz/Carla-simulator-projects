import numpy as np
from casadi import *
import casadi

# Define the MPC Controller Class
class MPCController:
    def __init__(self, N, dt, L):
        self.N = N 
        self.dt = dt  
        self.L = L 

        self.delta_max = np.radians(25)  
        self.a_max = 1.0  
        self.a_min = -1.0  

    def solve(self, x0, ref_traj):
        # x0: Current state [x, y, psi, v]
        # ref_traj: Reference trajectory over the horizon (4 x N+1)

        opti = casadi.Opti()

        X = opti.variable(4, self.N + 1)  # States: x, y, psi, v
        U = opti.variable(2, self.N)    

        x0_param = opti.parameter(4)
        ref = opti.parameter(4, self.N + 1)

        # Initial condition constraint
        opti.subject_to(X[:, 0] == x0_param)

        # Dynamics constraints
        for k in range(self.N):
            x_k = X[0, k]
            y_k = X[1, k]
            psi_k = X[2, k]
            v_k = X[3, k]
            delta_k = U[0, k]
            a_k = U[1, k]

            x_next = X[0, k + 1]
            y_next = X[1, k + 1]
            psi_next = X[2, k + 1]
            v_next = X[3, k + 1]

            # Vehicle model equations
            x_k_next = x_k + v_k * casadi.cos(psi_k) * self.dt
            y_k_next = y_k + v_k * casadi.sin(psi_k) * self.dt
            psi_k_next = psi_k + (v_k / self.L) * casadi.tan(delta_k) * self.dt
            v_k_next = v_k + a_k * self.dt

            # Add model constraints
            opti.subject_to(x_next == x_k_next)
            opti.subject_to(y_next == y_k_next)
            opti.subject_to(psi_next == psi_k_next)
            opti.subject_to(v_next == v_k_next)

            # Input constraints
            opti.subject_to(opti.bounded(-self.delta_max, delta_k, self.delta_max))
            opti.subject_to(opti.bounded(self.a_min, a_k, self.a_max))

        # costs functiomn
        J = 0

        # weights
        Q = np.diag([1.0, 1.0, 0.08, 10.5])   # state weights
        R = np.diag([1.0, 2.0])            # input wiehts
        
        # Q = np.diag([10.0, 10.0, 1.0, 1.0]) 
        # R = np.diag([0.1, 0.1])   

        for k in range(self.N):
            state_error = X[:, k] - ref[:, k]
            control_input = U[:, k]
            J += state_error.T @ Q @ state_error + control_input.T @ R @ control_input

        state_error = X[:, self.N] - ref[:, self.N]
        J += state_error.T @ Q @ state_error

                   
        Rd = np.diag([100.0, 10.0])  

        u_prev = opti.parameter(2, 1)

        # Modify the cost function to include rate of change of inputs
        # for k in range(self.N):
        #     state_error = X[:, k] - ref[:, k]
        #     control_input = U[:, k]
        #     J += state_error.T @ Q @ state_error + control_input.T @ R @ control_input
        #     if k > 0:
        #         delta_u = U[:, k] - U[:, k - 1]
        #     else:
        #         delta_u = U[:, k] - u_prev
        #     J += delta_u.T @ Rd @ delta_u

        # print("cost matrix")
        # print(J)

        opti.minimize(J)

        opti.set_value(x0_param, x0)
        opti.set_value(ref, ref_traj)

        opti.solver('ipopt', {'print_time': False, 'ipopt': {'print_level': 0}})

        try:
            sol = opti.solve()
            delta_opt = sol.value(U[0, :])
            a_opt = sol.value(U[1, :])

            return delta_opt[0], a_opt[0]
        except RuntimeError:
            print("MPC solver failed to find a solution.")
            return 0.0, 0.0
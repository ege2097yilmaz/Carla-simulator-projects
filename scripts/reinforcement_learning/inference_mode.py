import numpy as np
import pickle, carla
from collections import defaultdict
from enviroment import CarlaParkingEnv

class QLearningAgent:
    def __init__(self, state_bins, action_bins, q_table_path="q_table.pkl"):
        self.state_bins = state_bins
        self.action_bins = action_bins
        self.q_table_path = q_table_path
        self.q_table = self.load_q_table()

    def discretize_state(self, state):
        """Discretize continuous state into bins."""
        x, y = state
        x_bin = np.digitize(x, self.state_bins[0])
        y_bin = np.digitize(y, self.state_bins[1])
        return (x_bin, y_bin)

    def load_q_table(self):
        """Load Q-table from file."""
        with open(self.q_table_path, 'rb') as f:
            q_table = pickle.load(f)
        print(f"Q-table loaded from {self.q_table_path}")
        return defaultdict(lambda: np.zeros(len(self.action_bins)), q_table)

    def choose_action(self, state):
        """Choose the best action based on the Q-table."""
        return np.argmax(self.q_table[state])

# Control Loop
if __name__ == "__main__":
    # Initialize CARLA environment and Q-learning agent
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    goal = [-50, 35]  # Define goal location
    env = CarlaParkingEnv(client, goal)

    # Define discretization bins (must match the training settings)
    state_bins = [
        np.linspace(-1000, 1000, 20),  # Bins for x-coordinate
        np.linspace(-1000, 1000, 20)  # Bins for y-coordinate
    ]
    action_bins = [
        (throttle, steer)
        for throttle in np.linspace(0, 1, 5)  # Discretized throttle values
        for steer in np.linspace(-1, 1, 5)   # Discretized steering values
    ]

    agent = QLearningAgent(state_bins, action_bins)

    # Control the vehicle using the trained Q-table
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.visualize_goal_with_pointer()
        state_discrete = agent.discretize_state(state)
        action_index = agent.choose_action(state_discrete)
        action = action_bins[action_index]

        state, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Goal reached! Total Reward: {total_reward}")
    env.close()

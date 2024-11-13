import numpy as np
import random, os
from collections import defaultdict
import pickle

class QLearningAgent:
    def __init__(self, state_bins, action_bins, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, q_table_path="q_table.pkl"):
        self.state_bins = state_bins
        self.action_bins = action_bins
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        # self.q_table = defaultdict(lambda: np.zeros(len(action_bins)))
        self.q_table_path = q_table_path
        self.q_table = self.load_q_table()

    def discretize_state(self, state):
        """Discretize continuous state into bins."""
        x, y = state
        x_bin = np.digitize(x, self.state_bins[0])
        y_bin = np.digitize(y, self.state_bins[1])
        return (x_bin, y_bin)
    
    def get_q_table(self):
        return self.q_table

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.action_bins)))
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """Update the Q-table using the Q-learning formula."""
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

        # Decay epsilon
        if done:
            self.epsilon *= self.epsilon_decay
    
    def save_q_table(self):
        """Save Q-table to file."""
        with open(self.q_table_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {self.q_table_path}")

    def load_q_table(self):
        """Load Q-table from file if it exists."""
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, 'rb') as f:
                q_table = pickle.load(f)
            print(f"Q-table loaded from {self.q_table_path}")
            return defaultdict(lambda: np.zeros(len(self.action_bins)), q_table)
        else:
            print("No existing Q-table found. Starting with a new table.")
            return defaultdict(lambda: np.zeros(len(self.action_bins)))

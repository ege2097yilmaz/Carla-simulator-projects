import carla
import numpy as np
from q_learning import QLearningAgent
from enviroment import CarlaParkingEnv
import matplotlib.pyplot as plt



def visualize_state_action_heatmap(agent, state_bin):
    """
    Visualizes the state-action value heatmap for a given state.

    Args:
        agent: The QLearningAgent object.
        state_bin: The discretized state (e.g., (x_bin, y_bin)) to analyze.
    """
    # Get Q-values for the specified state
    q_values = agent.q_table[state_bin]

    # Reshape Q-values into a grid for throttle and steering
    num_throttle = 5  # Number of throttle bins (from action_bins definition)
    num_steer = 5     # Number of steer bins (from action_bins definition)
    q_grid = q_values.reshape((num_throttle, num_steer))

    # Create a heatmap
    plt.imshow(q_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Q-value')
    plt.xlabel('Steering Index')
    plt.ylabel('Throttle Index')
    plt.title(f'State-Action Heatmap for State {state_bin}')
    plt.xticks(range(num_steer), [f"Steer {i}" for i in range(num_steer)])
    plt.yticks(range(num_throttle), [f"Throttle {i}" for i in range(num_throttle)])
    plt.show()

if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    settings = world.get_settings()
    # settings.synchronous_mode = True  # Enable synchronous mode
    settings.fixed_delta_seconds = 0.05  # Increase simulation step time
    world.apply_settings(settings)

    # Define goal location
    goal = [-50, 35]
    episode_rewards = []
    successes = []  
    state_bin_example = (10, 10)

    env = CarlaParkingEnv(client, goal)

    # Define discretization bins
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
    
    episodes = 500
    max_steps_per_episode = 350

    for episode in range(episodes):
        state = env.reset()
        state_discrete = agent.discretize_state(state)
        total_reward = 0
        success = False

        for step in range(max_steps_per_episode):
            action_index = agent.choose_action(state_discrete)
            action = action_bins[action_index]

            next_state, reward, done, _ = env.step(action)
            next_state_discrete = agent.discretize_state(next_state)

            agent.update(state_discrete, action_index, reward, next_state_discrete, done)

            state_discrete = next_state_discrete
            total_reward += reward

            if done:
                success = reward == 100.0
                successes.append(success)
                break
        
        # Save Q-table after each episode
        agent.save_q_table()
        episode_rewards.append(total_reward)

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        

        # # Visualize every 10 episodes
        # if (episode + 1) % 10 == 0:
        #     visualize_state_action_heatmap(agent, state_bin_example)

        #     plt.plot(episode_rewards)
        #     plt.xlabel('Episodes')
        #     plt.ylabel('Total Reward')
        #     plt.title('Training Progress: Total Rewards per Episode')
        #     plt.show()

        #     if len(successes) > 0:
        #         # Calculate and plot success rate
        #         success_rate = np.cumsum(successes[:episode + 1]) / np.arange(1, episode + 2)
        #         plt.figure()
        #         plt.plot(success_rate)
        #         plt.xlabel('Episodes')
        #         plt.ylabel('Success Rate')
        #         plt.title('Training Progress: Success Rate')
        #         plt.show()

    env.close()
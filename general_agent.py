import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class GeneralAgent:
    def __init__(self, env_name: str, learning_rate: float, initial_epsilon: float,
                 epsilon_decay: float, final_epsilon: float, discount_factor: float = 0.95,
                 n_episodes: int = 100_000, verbose: bool = False, render_mode: str = "rgb_array", algorithm: str = 'q_learning'):
        """Initialize a Reinforcement Learning agent for various environments."""
        self.env = gym.make(env_name, render_mode=render_mode)
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, deque_size=n_episodes)
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.n_episodes = n_episodes
        self.verbose = verbose
        self.training_error = []
        self.algorithm = algorithm  # Algorithm to use ('sarsa' or 'q_learning')


    def get_action(self, obs):
        """Choose an action based on the current observation."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        """Updates the Q-value of an action based on the observed transition."""
        if self.algorithm == 'q_learning':
            future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        elif self.algorithm == 'sarsa':
            next_action = self.get_action(next_obs) if not terminated else 0
            future_q_value = self.q_values[next_obs][next_action]

        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        self.q_values[obs][action] += self.learning_rate * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce epsilon gradually to shift from exploration to exploitation."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def train(self):
        """Conduct training over a set number of episodes."""
        for episode in tqdm(range(self.n_episodes)):
            curr_observation, info = self.env.reset()
            while True:
                if self.verbose:
                    self.print_verbose_info(curr_observation)

                action = self.get_action(curr_observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                is_terminal = terminated or truncated

                self.update(curr_observation, action, reward, terminated, next_observation)
                curr_observation = next_observation

                if is_terminal:
                    break

            self.decay_epsilon()

    def print_verbose_info(self, observation):
        """Prints detailed information about the current state for debugging."""
        print("Current observation:", observation)
        # Depending on the environment, additional debug information can be added here
        print("Next observation:", observation)  # Adjusted to show relevant debug info
        print("-" * 20)

    def get_folder_name(self):
        """Generates a consistent folder name based on agent parameters."""
        env_name = self.env.unwrapped.spec.id
        # Including decay rate and number of episodes in the folder name
        folder_name = f"{env_name}_lr{self.learning_rate:.2f}_eps{self.initial_epsilon:.2f}_epsdec{self.epsilon_decay:.2f}_epsfin{self.final_epsilon:.2f}_disc{self.discount_factor:.3f}_nep{self.n_episodes}_alg{self.algorithm}"
        folder_path = os.path.join("./figures", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path


    def plot_training_info(self, rolling_length=1000):
        """Plot training information for rewards, episode lengths, and training errors."""

        folder_path = self.get_folder_name()
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
        # Rewards
        reward_moving_average = np.convolve(np.array(self.env.return_queue).flatten(), np.ones(rolling_length), mode="valid") / rolling_length
        axs[0].set_title("Episode Reward")
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        # Lengths
        length_moving_average = np.convolve(np.array(self.env.length_queue).flatten(), np.ones(rolling_length), mode="same") / rolling_length
        axs[1].set_title("Episode Length")
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        # Errors
        training_error_moving_average = np.convolve(np.array(self.training_error), np.ones(rolling_length), mode="same") / rolling_length
        axs[2].set_title("Training Error")
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, "training_info.pdf"))

        plt.close(fig)

    def visualize_policy(self):
        """Visualizes the policy derived from the Q-values."""

        env_name = self.env.unwrapped.spec.id
        folder_path = self.get_folder_name()

        if env_name in ['FrozenLake-v1', 'FrozenLake8x8-v1']:
            # Assuming a grid layout for these environments
            side_length = int(np.sqrt(self.env.observation_space.n))
            policy_grid = np.array([np.argmax(self.q_values[state]) if state in self.q_values else -1
                                    for state in range(self.env.observation_space.n)])
            policy_grid = policy_grid.reshape((side_length, side_length))
            plt.figure(figsize=(8, 8))
            plt.imshow(policy_grid, cmap="viridis")
            plt.colorbar()
            plt.title(f"Policy Map for {env_name}")
            plt.xlabel("Grid X")
            plt.ylabel("Grid Y")

            plt.savefig(os.path.join(folder_path, "policy_visualization.pdf"))
            plt.close()

        elif env_name == 'CliffWalking-v0':
            side_length = int(np.sqrt(self.env.observation_space.n))
            policy_grid = np.array([np.argmax(self.q_values[state]) if state in self.q_values else -1
                                    for state in range(self.env.observation_space.n)])
            policy_grid = policy_grid.reshape((6, 8))
            plt.figure(figsize=(8, 8))
            plt.imshow(policy_grid, cmap="viridis")
            plt.colorbar()
            plt.title(f"Policy Map for {env_name}")
            plt.xlabel("Grid X")
            plt.ylabel("Grid Y")

            plt.savefig(os.path.join(folder_path, "policy_visualization.pdf"))
            plt.close()

        elif env_name == 'Taxi-v3':
            # Taxi state space is more complex, decomposing state for visualization
            num_states = self.env.observation_space.n
            policy_grid = np.zeros((5,5,5))
            for state in range(num_states):
                decoded = list(self.env.unwrapped.decode(state))
                row, col, pass_idx, dest_idx = decoded
                best_action = np.argmax(self.q_values[state])
                policy_grid[row, col, pass_idx] = best_action
            plt.figure(figsize=(10, 10))
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.imshow(policy_grid[:, :, i], cmap="viridis")
                plt.title(f"Passenger {i} Policy")
                plt.colorbar()
            plt.tight_layout()

            plt.savefig(os.path.join(folder_path, "policy_visualization.pdf"))
            plt.close()

        elif env_name == 'Blackjack-v1':
            # Blackjack visualization (simplified example, considering only totals and ace presence)
            player_range = np.arange(12, 22)  # Player's total range considered
            dealer_range = np.arange(1, 11)   # Dealer's showing card
            usable_ace = [False, True]        # Usable ace scenarios
            policy_grid = np.zeros((len(player_range), len(dealer_range), 2))
            for i, player in enumerate(player_range):
                for j, dealer in enumerate(dealer_range):
                    for k, ace in enumerate(usable_ace):
                        state = (player, dealer, ace)
                        if state in self.q_values:
                            policy_grid[i, j, k] = np.argmax(self.q_values[state])
            plt.figure(figsize=(12, 6))
            for i in range(2):
                plt.subplot(1, 2, i + 1)
                plt.imshow(policy_grid[:, :, i], cmap="viridis", aspect='auto')
                plt.title(f"Policy with {'usable' if i == 1 else 'no'} ace")
                plt.xlabel("Dealer showing")
                plt.ylabel("Player total")
                plt.xticks(ticks=np.arange(len(dealer_range)), labels=dealer_range)
                plt.yticks(ticks=np.arange(len(player_range)), labels=player_range)
                plt.colorbar()
            plt.tight_layout()


            plt.savefig(os.path.join(folder_path, "policy_visualization.pdf"))

            plt.close()
        else:
            print("Environment not supported for direct visualization.")


if __name__ == "__main__":
    agent = GeneralAgent(
        env_name='Taxi-v3',
        learning_rate=0.01,
        initial_epsilon=0.9,
        epsilon_decay=0.01,
        final_epsilon=0.1,
        discount_factor=0.95,
        n_episodes=10000,
        verbose=False,
        algorithm='sarsa'  # Choose 'sarsa' or 'q_learning'
    )
    agent.train()
    agent.plot_training_info()
    agent.visualize_policy()

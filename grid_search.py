import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Assuming GeneralAgent class is defined in general_agent.py
from general_agent import GeneralAgent

# Define the hyperparameters grid
learning_rates = [0.01, 0.1]
initial_epsilons = [0.2, 0.7, 1.0]
epsilon_decays = [0.01, 0.02]
final_epsilons = [0.01, 0.05]
discount_factors = [0.9, 0.95, 0.99]
n_episodes_list = [20_000, 100_000]
toy_text_envs = ['Blackjack-v1', 'CliffWalking-v0', 'FrozenLake-v1', 'Taxi-v3']

# Store overall results to calculate averages
results = defaultdict(list)

# Iterate over all combinations of hyperparameters and environments
for env_name in toy_text_envs:
    for lr in learning_rates:
        for init_eps in initial_epsilons:
            for eps_decay in epsilon_decays:
                for final_eps in final_epsilons:
                    for discount in discount_factors:
                        for n_episodes in n_episodes_list:
                            average_rewards = []
                            average_lengths = []
                            std_rewards = []
                            
                            # Execute each configuration 5 times
                            for trial in range(5):
                                # Create the agent
                                agent = GeneralAgent(env_name=env_name, learning_rate=lr, initial_epsilon=init_eps,
                                                     epsilon_decay=eps_decay, final_epsilon=final_eps, 
                                                     discount_factor=discount, n_episodes=n_episodes, verbose=False)
                                agent.train()

                                # Evaluate performance
                                n_eval = 1000 if n_episodes == 20_000 else 5000
                                metrics = agent.evaluate_performance(last_n_episodes=n_eval)
                                
                                average_rewards.append(metrics['average_reward'])
                                average_lengths.append(metrics['average_length'])
                                std_rewards.append(metrics['std_reward'])
                            
                            # Calculate averages of the metrics
                            avg_reward = np.mean(average_rewards)
                            avg_length = np.mean(average_lengths)
                            avg_std_reward = np.mean(std_rewards)
                            
                            # Save the training information and the policy
                            agent.plot_training_info()
                            agent.visualize_policy()
                            
                            # Record results
                            results_key = (env_name, lr, init_eps, eps_decay, final_eps, discount, n_episodes)
                            results[results_key].extend([avg_reward, avg_length, avg_std_reward])

                            print(f"Average metrics for {env_name} | LR={lr}, InitEps={init_eps}, EpsDecay={eps_decay}, FinalEps={final_eps}, Discount={discount}, Episodes={n_episodes}:")
                            print(f"Avg. Reward: {avg_reward}, Avg. Length: {avg_length}, Std. Reward: {avg_std_reward}")

print("Grid search completed.")


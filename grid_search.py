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
initial_epsilons = [0.2, 0.7, 0.95]  # Updated range of initial epsilon values
epsilon_decays = [0.01, 0.02]
final_epsilons = [0.01, 0.05]
discount_factors = [0.9, 0.95, 0.99]  # Discount factors
n_episodes_list = [20_000, 100_000]

# Add all toy-text environments
toy_text_envs = ['Blackjack-v1', 'CliffWalking-v0', 'FrozenLake-v1', 'Taxi-v3']

# Iterate over all combinations of hyperparameters and environments
for env_name in toy_text_envs:
    for lr in learning_rates:
        for init_eps in initial_epsilons:
            for eps_decay in epsilon_decays:
                for final_eps in final_epsilons:
                    for discount in discount_factors:
                        for n_episodes in n_episodes_list:
                            # Create the agent with the current set of hyperparameters
                            agent = GeneralAgent(env_name=env_name, learning_rate=lr, initial_epsilon=init_eps,
                                                 epsilon_decay=eps_decay, final_epsilon=final_eps, 
                                                 discount_factor=discount, n_episodes=n_episodes, verbose=False)
                            
                            # Train the agent
                            print(f"Training {env_name} with LR={lr}, InitEps={init_eps}, EpsDecay={eps_decay}, FinalEps={final_eps}, Discount={discount}, Episodes={n_episodes}")
                            agent.train()

                            # Save the training information and the policy
                            agent.plot_training_info()
                            agent.visualize_policy()

print("Grid search completed.")


# Model-Free Tabular Methods

The `GeneralAgent` class provides a framework for training and visualizing agent performance in various environments from the Gymnasium Toy-Text collection. This class simplifies the process of specifying hyperparameters, training the agent, and evaluating its performance.

## Usage

First, initialize the `GeneralAgent` with the desired environment and algorithm parameters:

```python
from general_agent import GeneralAgent

agent = GeneralAgent(
    env_name='Taxi-v3',           # Environment ID from Gymnasium
    learning_rate=0.01,           # Step size for updating estimates
    initial_epsilon=0.9,          # Initial exploration rate
    epsilon_decay=0.01,           # Decay rate of epsilon per episode
    final_epsilon=0.1,            # Minimum value of epsilon after decay
    discount_factor=0.95,         # Discount factor for future rewards
    n_episodes=100,               # Number of episodes for training
    verbose=False,                # Toggle to display detailed logs
    algorithm='sarsa'             # Algorithm type: 'sarsa' or 'q_learning'
)
```

### Training

To train the agent on the specified environment:

```python
agent.train()
```

### Visualization

Visualize the training information and the policy after training:

```python
agent.plot_training_info()   # Plot statistics like cumulative rewards per episode
agent.visualize_policy()     # Show the learned policy in a human-readable format
```

### Performance Evaluation

Evaluate the agent's performance based on the last `n` episodes:

```python
performance_metrics = agent.evaluate_performance(last_n_episodes=100)
print("Performance metrics:", performance_metrics)
```

This will display the collected performance metrics such as average rewards, number of steps per episode, and other relevant statistics.


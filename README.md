# Q-Learning Robot Navigation with Dyna-Q

A Python implementation of Q-Learning reinforcement learning algorithm with optional Dyna-Q model-based planning for robot navigation in grid-world environments.

## Overview

This project provides a flexible Q-Learning agent that can learn optimal navigation policies through trial and error. The agent builds a Q-table mapping state-action pairs to expected rewards, enabling it to navigate through grid environments while avoiding obstacles and reaching goals efficiently.

### Key Features

- **Standard Q-Learning**: Classic temporal difference learning algorithm
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Dyna-Q Planning**: Model-based learning for significantly faster convergence
- **Configurable Parameters**: Easily tune learning rate, discount factor, and exploration
- **Grid-World Navigation**: Designed for discrete state-action environments
- **Policy Extraction**: Methods to visualize and analyze learned behavior

## Installation

### Requirements

- Python 3.6+
- NumPy

```bash
pip install numpy
```

### Setup

Simply download or clone the repository:

```bash
git clone <repository-url>
cd qlearning-navigation
```

## Usage

### Basic Example

```python
from QLearner import QLearner

# Create a Q-Learning agent
learner = QLearner(
    num_states=100,      # 10x10 grid world
    num_actions=4,       # N, E, S, W movements
    alpha=0.2,           # Learning rate
    gamma=0.9,           # Discount factor
    rar=0.5,             # Initial exploration rate
    radr=0.99,           # Exploration decay
    dyna=200,            # Dyna-Q planning steps
    verbose=False
)

# Initialize episode
initial_action = learner.querysetstate(start_state)

# Learning loop
for step in range(max_steps):
    # Environment determines next state and reward
    next_state, reward = environment.step(initial_action)
    
    # Agent learns and selects next action
    next_action = learner.query(next_state, reward)
    
    if environment.is_terminal(next_state):
        break
```

### State Representation

States are encoded as integers using the formula:
```
state = row * 10 + column
```

For a 10x10 grid, valid states range from 0 to 99.

### Action Space

Four discrete actions are available:
- **0**: North (move up)
- **1**: East (move right)
- **2**: South (move down)
- **3**: West (move left)

## Algorithm Details

### Q-Learning Update Rule

The agent updates Q-values using the temporal difference learning rule:

```
Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
```

Where:
- `α` (alpha): Learning rate
- `γ` (gamma): Discount factor
- `r`: Immediate reward
- `s, a`: Current state and action
- `s'`: Next state

### Dyna-Q Planning

When enabled (dyna > 0), the agent:

1. **Learns from real experience**: Updates Q-table with actual environment interactions
2. **Builds an internal model**: Stores transition and reward information
3. **Simulates experiences**: Samples from memory to generate additional training data
4. **Plans ahead**: Updates Q-table with simulated experiences for faster learning

This hybrid approach combines the sample efficiency of model-based methods with the simplicity of model-free Q-learning.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_states` | int | 100 | Total number of states in environment |
| `num_actions` | int | 4 | Number of available actions |
| `alpha` | float | 0.2 | Learning rate (0.0-1.0) |
| `gamma` | float | 0.9 | Discount factor for future rewards (0.0-1.0) |
| `rar` | float | 0.5 | Random action rate (exploration probability) |
| `radr` | float | 0.99 | Random action decay rate per step |
| `dyna` | int | 0 | Number of planning steps per real experience |
| `verbose` | bool | False | Enable debug output |

### Parameter Tuning Tips

- **Higher alpha** (e.g., 0.3-0.5): Faster learning but less stable
- **Lower alpha** (e.g., 0.1-0.2): More stable but slower convergence
- **Higher gamma** (e.g., 0.95-0.99): Values long-term rewards more
- **Lower gamma** (e.g., 0.8-0.9): Focuses on immediate rewards
- **Dyna = 0**: Pure model-free Q-learning
- **Dyna = 50-200**: Good balance for most problems
- **Dyna > 200**: Faster learning but higher computational cost

## API Reference

### Core Methods

#### `querysetstate(s)`
Initialize the agent at the start of an episode.
- **Args**: `s` - Starting state
- **Returns**: Initial action

#### `query(s_prime, r)`
Main learning method called after each action.
- **Args**: 
  - `s_prime` - New state after action
  - `r` - Reward received
- **Returns**: Next action to take

### Analysis Methods

#### `get_q_table()`
Returns a copy of the Q-table for analysis.

#### `get_policy()`
Extracts the greedy policy (best action per state).

#### `get_value_function()`
Returns the state value function V(s) = max_a Q(s,a).

#### `reset_exploration(rar=0.5)`
Resets the exploration rate (useful between episodes).

## Project Structure

```
qlearning-navigation/
├── QLearner.py          # Main Q-Learning implementation
├── README.md            # This file
└── LICENSE              # MIT License
```

## Performance Considerations

- **Memory**: O(num_states × num_actions) for Q-table
- **With Dyna**: Additional O(num_states × num_actions) for model storage
- **Time per step**: O(1) for Q-learning, O(dyna) for planning
- **Convergence**: Typically faster with Dyna enabled (50-200 steps recommended)

## License

MIT License - see LICENSE file for details

## Author

Timothy Bradford

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
- Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292
- Sutton, R. S. (1990). Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. *Proceedings of the Seventh International Conference on Machine Learning*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Future Enhancements

- [ ] Support for stochastic environments
- [ ] Double Q-Learning implementation
- [ ] Function approximation for large state spaces
- [ ] Visualization tools for policy and value functions
- [ ] Priority sweeping for Dyna-Q
- [ ] Experience replay buffer

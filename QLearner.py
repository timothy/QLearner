"""
Q-Learning Robot Navigation with Dyna-Q Extension

This module implements a Q-Learning algorithm with Dyna-Q model-based planning
for robot navigation tasks. The agent learns to navigate through a grid world
environment to reach a goal while avoiding obstacles.

Author: Timothy Bradford
License: MIT License

Key Features:
- Standard Q-Learning with epsilon-greedy exploration
- Dyna-Q model-based planning for faster learning
- Configurable learning parameters
- Support for grid-world navigation tasks
"""

import random as rand
import numpy as np


class QLearner:
    """
    Q-Learning agent with optional Dyna-Q model-based planning.
    
    The agent learns optimal navigation policies through trial and error,
    building a Q-table that maps state-action pairs to expected rewards.
    
    State representation: row_location * 10 + column_location
    Actions: 0=North, 1=East, 2=South, 3=West
    """
    
    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        """
        Initialize the Q-Learning agent.
        
        Args:
            num_states: Number of possible states in the environment
            num_actions: Number of available actions (typically 4 for grid navigation)
            alpha: Learning rate (0.0-1.0), controls how quickly Q-values are updated
            gamma: Discount factor (0.0-1.0), weights importance of future rewards
            rar: Random action rate, probability of exploration vs exploitation
            radr: Random action decay rate, reduces exploration over time
            dyna: Number of planning steps per real experience (0 = disabled)
            verbose: Enable debug output
        """
        # Core parameters
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.num_actions = num_actions
        
        # Q-table: stores learned action values for each state
        self.Q = np.zeros((num_states, num_actions))
        
        # Current state and action
        self.s = 0
        self.a = 0
        
        # Dyna-Q model components
        if dyna > 0:
            # Transition model: T[s,a] = s' (deterministic)
            self.T = np.full((num_states, num_actions), -1, dtype=np.int32)
            # Reward model: R[s,a] = expected reward
            self.R = np.zeros((num_states, num_actions))
            # Track which state-action pairs have been visited
            self.visited_states = set()
            self.visited_state_actions = set()
    
    def querysetstate(self, s):
        """
        Set the agent's state and return initial action without learning.
        
        Used to initialize the agent at the start of an episode.
        
        Args:
            s: The starting state
            
        Returns:
            The selected action for the given state
        """
        self.s = s
        action = self._select_action(s)
        self.a = action
        
        if self.verbose:
            print(f"Initial state: s={s}, action={action}")
        
        return action
    
    def query(self, s_prime, r):
        """
        Core learning method: update Q-table and return next action.
        
        Updates the Q-table based on the experience tuple (s, a, s', r),
        optionally performs Dyna planning, and selects the next action.
        
        Args:
            s_prime: The new state after taking action a from state s
            r: The reward received for the transition
            
        Returns:
            The next action to take from state s_prime
        """
        # Update Q-table with real experience
        self._update_q_table(self.s, self.a, s_prime, r)
        
        # Perform Dyna-Q planning if enabled
        if self.dyna > 0:
            self._dyna_planning(self.s, self.a, s_prime, r)
        
        # Select next action
        a_prime = self._select_action(s_prime)
        
        # Decay exploration rate
        self.rar *= self.radr
        
        if self.verbose:
            print(f"Experience: s={self.s}, a={self.a}, s'={s_prime}, r={r:.2f}, next_a={a_prime}")
        
        # Update current state and action
        self.s = s_prime
        self.a = a_prime
        
        return a_prime
    
    def _select_action(self, state):
        """
        Select an action using epsilon-greedy strategy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if rand.uniform(0, 1) < self.rar:
            # Exploration: random action
            return rand.randint(0, self.num_actions - 1)
        else:
            # Exploitation: best known action
            return int(np.argmax(self.Q[state]))
    
    def _update_q_table(self, s, a, s_prime, r):
        """
        Update Q-value using the Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        
        Args:
            s: Previous state
            a: Action taken
            s_prime: Resulting state
            r: Reward received
        """
        current_q = self.Q[s, a]
        max_future_q = np.max(self.Q[s_prime])
        
        # TD target
        target = r + self.gamma * max_future_q
        
        # TD error
        td_error = target - current_q
        
        # Update Q-value
        self.Q[s, a] = current_q + self.alpha * td_error
    
    def _dyna_planning(self, s, a, s_prime, r):
        """
        Perform Dyna-Q model-based planning.
        
        Updates the internal model of the environment and uses it
        to generate simulated experiences for additional learning.
        
        Args:
            s: State
            a: Action
            s_prime: Next state
            r: Reward
        """
        # Update the model with real experience
        self._update_model(s, a, s_prime, r)
        
        # Perform planning steps
        for _ in range(self.dyna):
            # Sample from previously visited state-action pairs
            sim_s, sim_a = self._sample_state_action()
            
            if sim_s is not None:
                # Use model to predict next state and reward
                sim_s_prime, sim_r = self._model_predict(sim_s, sim_a)
                
                if sim_s_prime is not None:
                    # Update Q-table with simulated experience
                    self._update_q_table(sim_s, sim_a, sim_s_prime, sim_r)
    
    def _update_model(self, s, a, s_prime, r):
        """
        Update the internal model of the environment.
        
        Args:
            s: State
            a: Action
            s_prime: Next state
            r: Reward
        """
        # Deterministic transition model
        self.T[s, a] = s_prime
        
        # Update reward model (running average could be used for stochastic rewards)
        self.R[s, a] = r
        
        # Track visited states and state-action pairs
        self.visited_states.add(s)
        self.visited_state_actions.add((s, a))
    
    def _sample_state_action(self):
        """
        Sample a random state-action pair from previously visited ones.
        
        Returns:
            Tuple of (state, action) or (None, None) if no history exists
        """
        if not self.visited_state_actions:
            return None, None
        
        # Random sampling from visited state-action pairs
        s, a = rand.choice(list(self.visited_state_actions))
        return s, a
    
    def _model_predict(self, s, a):
        """
        Use the model to predict next state and reward.
        
        Args:
            s: State
            a: Action
            
        Returns:
            Tuple of (next_state, reward) or (None, 0) if not in model
        """
        s_prime = self.T[s, a]
        
        # Check if this state-action has been visited
        if s_prime == -1:
            return None, 0
        
        r = self.R[s, a]
        return s_prime, r
    
    def get_q_table(self):
        """
        Get a copy of the Q-table for analysis or visualization.
        
        Returns:
            Copy of the Q-table
        """
        return self.Q.copy()
    
    def get_policy(self):
        """
        Extract the learned policy from the Q-table.
        
        Returns:
            Array where each element is the best action for that state
        """
        return np.argmax(self.Q, axis=1)
    
    def get_value_function(self):
        """
        Get the state value function V(s) = max_a Q(s,a).
        
        Returns:
            Array of state values
        """
        return np.max(self.Q, axis=1)
    
    def reset_exploration(self, rar=0.5):
        """
        Reset the exploration rate (useful for new episodes).
        
        Args:
            rar: New random action rate
        """
        self.rar = rar


if __name__ == "__main__":
    # Example usage
    print("Q-Learning Agent with Dyna-Q Planning")
    print("-" * 40)
    
    # Create a simple Q-learner
    learner = QLearner(
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=200,  # Enable Dyna-Q with 200 planning steps
        verbose=False
    )
    
    print(f"Agent initialized with:")
    print(f"  States: 100")
    print(f"  Actions: 4 (N, E, S, W)")
    print(f"  Learning rate: 0.2")
    print(f"  Discount factor: 0.9")
    print(f"  Dyna planning steps: 200")

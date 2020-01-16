'''
Agents for cartpole
'''
import numpy as np
import math

class Agent(object):
    """
    generic class for agents
    """
    def __init__(self, action_set, reward_function):
        self.action_set = action_set
        self.reward_function = reward_function
        self.cummulative_reward = 0

    def __str__(self):
        pass

    def reset_cumulative_reward(self):
        self.cummulative_reward = 0

    def update_buffer(self, observation_history, action_history):
        pass

    def learn_from_buffer(self):
        pass

    def act(self, observation_history, action_history):
        pass

    def get_episode_reward(self, observation_history, action_history):
        tau = len(action_history)
        reward_history = np.zeros(tau)
        for t in range(tau):
            reward_history[t] = self.reward_function(
                observation_history[:t+2], action_history[:t+1])
            # print(reward_history[t])
        return reward_history

    def _random_argmax(self, action_values):
        argmax_list = np.where(action_values==np.max(action_values))[0]
        return self.action_set[argmax_list[np.random.randint(argmax_list.size)]]

    def _epsilon_greedy_action(self, action_values, epsilon):
        if np.random.random() < 1- epsilon:
            return self._random_argmax(action_values)
        else:
            return np.random.choice(self.action_set, 1)[0]

    def _boltzmann_action(self, action_values, beta):
        action_values = action_values - max(action_values)
        action_probabilities = np.exp(action_values / beta)
        action_probabilities /= np.sum(action_probabilities)
        return np.random.choice(self.action_set, 1, p=action_probabilities)[0]

    def _epsilon_boltzmann_action(self, action_values, epsilon):
        action_values = np.array(action_values)
        print('print',action_values)
        action_values = action_values - max(action_values)
        action_probabilities = np.exp(action_values / (np.exp(1)*epsilon))
        action_probabilities /= np.sum(action_probabilities)
        return np.random.choice(self.action_set, 1, p=action_probabilities)[0]

# -------------------------------------------------------
# Random Agent - picks an action uniformly at random
# -------------------------------------------------------

class RandomAgent(Agent):
    """
    selects actions uniformly at random from the action set
    """
    def __str__(self):
        return "random agent"

    def act(self, observation_history, action_history):
        return np.random.choice(self.action_set, 1)[0]

    def update_buffer(self, observation_history, action_history):
        reward_history = self.get_episode_reward(observation_history, action_history)
        self.cummulative_reward += np.sum(reward_history)

# ---------------------------------------------------------
# Tabular LSVI agent with epsilon Boltzmann
# ---------------------------------------------------------
class TabularLsviAgent(Agent):
    def __init__(self,
                 action_set,
                 reward_function,
                 epsilon,
                 num_iterations,
                 feature_extractor):
        Agent.__init__(self,action_set,reward_function)
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.feature_extractor = feature_extractor

        # buffer is a dictionary of lists
        # the key is a feature-action pair
        self.buffer = {(f, a): [] for f in self.feature_extractor.feature_space for a in self.action_set}
        self.Q = {key: 100 for key in self.buffer.keys()}

    def __str__(self):
        return 'tabular_lsvi_agent_with_eBoltzmann'

    def update_buffer(self, observation_history, action_history):
        reward_history = self.get_episode_reward(observation_history, action_history)
        self.cummulative_reward += np.sum(reward_history)
        tau = len(action_history)
        feature_history = [self.feature_extractor.get_feature(observation_history[:t + 1])
                           for t in range(tau + 1)]
        for t in range(tau - 1):
            new_key = (feature_history[t], action_history[t])
            new_item = (reward_history[t], feature_history[t + 1])
            self.buffer[new_key].append(new_item)
        done = observation_history[tau][1]
        if done:
            feat_next = None
        else:
            feat_next = feature_history[tau]

        new_key = (feature_history[tau - 1], action_history[tau - 1])
        new_item = (reward_history[tau - 1], feat_next)
        self.buffer[new_key].append(new_item)

    def learn_from_buffer(self):
        Q = {key: 0.0 for key in self.buffer.keys()}
        for n in range(self.num_iterations):
            for key in self.buffer.keys():
                #print("key",key)
                q = 0.0
                for transition in self.buffer[key]:
                    #print("transition",transition)
                    if transition[1] == None:
                        q += transition[0]
                    else:
                        v = max(Q[(transition[1], a)] for a in self.action_set)
                        #print("v", v)
                        q += transition[0] + v
                if len(self.buffer[key])>0:
                    Q[key] = q/(len(self.buffer[key]))
                else:
                    Q[key] = 0.0
        self.Q = Q

    def act(self, observation_history, action_history):
        feature = self.feature_extractor.get_feature(observation_history)
        return self._epsilon_boltzmann_action([self.Q[(feature, a)] for a in self.action_set]
                                              , self.epsilon)

# -------------------------------------------------------
# reward function
# -------------------------------------------------------
def cartpole_reward_function(observation_history, action_history,
    reward_type='height', move_cost=0.001):
    """
    If the reward type is 'height,' agent gets a reward of 1 + cosine of the
    pole angle per step. Agent also gets a bonus reward of 1 if pole is upright
    and still.
    If the reward type is 'sparse,' agent gets 1 if the pole is upright
    and still and if the cart is around the center.
    There is a small cost for applying force to the cart.
    """
    state, terminated = observation_history[-1]
    x, x_dot, theta, theta_dot = state
    action = action_history[-1]

    reward = - move_cost * np.abs(action - 1.)

    if not terminated:
        up = math.cos(theta) > 0.95
        still = np.abs(theta_dot) <= 1
        # centered = (np.abs(x) <= 1) and (np.abs(x_dot) <= 1)
        centered = (np.abs(x) <= 1) and (np.abs(x_dot) <= 1)
        # centered = True
        if reward_type == 'height':
            reward += math.cos(theta) + 1 + (up and still)

        elif reward_type == 'sparse':
            reward += (up and still and centered)

    return reward

# ----------------------------------------------------------------
# reward function for deep sea experiments
# ---------------------------------------------------------------
def deep_sea_reward(observation_history,action_history,horizon,treasure = True,move_cost = 0.01):
    state = observation_history[-1][0]
    prev_state = observation_history[-2][0]
    # horizontal, vertical = state

    reward =0
    if state[0]-prev_state[0] ==1:
    # for deepsea we penalize 'right' action
    # only along the diagonal (Example 8)
        if prev_state[0]==prev_state[1]:
            reward = -move_cost / (horizon * 1.0)
    if state[1] == horizon:
        if state[0] == horizon:
            if treasure:
                reward += 1
            else:
                reward += -1
    return reward

def deep_sea_reward_root(observation_history,action_history,horizon,treasure = True,move_cost = 0.01):
    state = observation_history[-1][0]
    prev_state = observation_history[-2][0]
    # horizontal, vertical = state

    reward =0
    if state[0]-prev_state[0] ==1:
    # for deepsea we penalize 'right' action
    # only along the diagonal (Example 8)
        if prev_state[0]==prev_state[1]:
            reward = -np.sqrt(move_cost / (horizon * 1.0))
    if state[1] == horizon:
        if state[0] == horizon:
            if treasure:
                reward += 1
            else:
                reward += -1
    return reward

import numpy as np
import random
import copy

class Environment(object):
    '''General RL environment'''

    def __init__(self):
        pass

    def reset(self):
        pass

    def advance(self, action):
        '''
        Moves one step in the environment.
        Args:
            action
        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        return 0, 0, 0

def make_riverSwim(epLen=20, nState=5):
    '''
    Makes the benchmark RiverSwim MDP.
    Args:
        NULL - works for default implementation
    Returns:
        riverSwim - Tabular MDP environment '''
    nAction = 2
    R_true = {}
    P_true = {}
    states = {}
    for s in range(nState):
        states[(s)] = 0.0
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (5/1000, 0)
    R_true[nState - 1, 1] = (1, 0)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.

    for s in range(1, nState - 1):
        P_true[s, 1][min(nState - 1, s + 1)] = 0.3
        P_true[s, 1][s] = 0.6
        P_true[s, 1][max(0, s-1)] = 0.1

    P_true[0, 1][0] = 0.3
    P_true[0, 1][1] = 0.7
    P_true[nState - 1, 1][nState - 1] = 0.9
    P_true[nState - 1, 1][nState - 2] = 0.1

    riverSwim = TabularMDP(nState, nAction, epLen)
    riverSwim.R = R_true
    riverSwim.P = P_true
    riverSwim.states = states
    riverSwim.reset()

    return riverSwim

class TabularMDP(Environment):
    '''
    Tabular MDP
    R - dict by (s,a) - each R[s,a] = (meanReward, sdReward)
    P - dict by (s,a) - each P[s,a] = transition vector size S
    '''

    def __init__(self, nState, nAction, epLen):
        '''
        Initialize a tabular episodic MDP
        Args:
            nState  - int - number of states
            nAction - int - number of actions
            epLen   - int - episode length
        Returns:
            Environment object
        '''

        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen

        self.timestep = 0
        self.state = 0

        # Now initialize R and P
        self.R = {}
        self.P = {}
        self.states = {}
        for state in range(nState):
            for action in range(nAction):
                self.R[state, action] = (1, 1)
                self.P[state, action] = np.ones(nState) / nState

    def reset(self):
        "Resets the Environment"
        self.timestep = 0
        self.state = 0

    def advance(self,action):
        '''
        Move one step in the environment
        Args:
        action - int - chosen action
        Returns:
        reward - double - reward
        newState - int - new state
        episodeEnd - 0/1 - flag for end of the episode
        '''
        if self.R[self.state, action][1] < 1e-9:
            # Hack for no noise
            reward = self.R[self.state, action][0]
        else:
            reward = np.random.normal(loc=self.R[self.state, action][0],
                                      scale=self.R[self.state, action][1])
        #print(self.state, action, self.P[self.state, action])
        newState = np.random.choice(self.nState, p=self.P[self.state, action])

        # Update the environment
        self.state = newState
        self.timestep += 1

        episodeEnd = 0
        if self.timestep == self.epLen:
            episodeEnd = 1
            #newState = None
            self.reset()

        return reward, newState, episodeEnd

    def argmax(self,b):
        #print(b)
        return np.random.choice(np.where(b == b.max())[0])

class deep_sea(Environment):
    '''
    Description:
        A deep sea environment, where a diver goes
        down and each time and she needs to make a
        decision to go left or right.
        environment terminates after fixed time step

    Observation:
        [horizontal position, vertical position]

    Actions:
        2 possible actions:
        0 - left
        1 - right

    Starting State:
        start at position 0, time step 0

    Episode termination:
        Env terminates after fixed number of time steps
    '''

    def __init__(self, num_steps):
        '''
        Inputs:
            num_steps: An integer that represents the size of the DeepSea environment
        '''
        self.num_steps = num_steps
        self.epLen = num_steps
        self.flip_mask = 2*np.random.binomial(1,0.5,(num_steps,num_steps))-1
        self.nAction = 2
        self.nState = pow(num_steps,2)
        self.epLen = num_steps
        self.R = {}
        self.states = {}
        for s in range(self.num_steps):
            for s_ in range(self.num_steps):
                self.R[(s,s_), 0] = (0, 0)
                self.R[(s,s_), 1] = (-0.01/self.nState, 0)
                self.states[(s,s_)] = 0
        self.R[(self.num_steps-1,self.num_steps-1),1] = (0.99,0)

    def name(self):
        return  "deep sea"

    def reset(self):
        self.state = (0,0)
        self.timestep = 0
        return copy.deepcopy(self.state)

    def advance(self,action):
        assert action in [0,1], "invalid action"
        self.state_prev = self.state
        step_horizontal = (2*action-1)
        horizontal = max(self.state[0] + step_horizontal, 0)
        vertical = self.state[1] + 1
        done =  bool(vertical == self.num_steps)
        self.state = (horizontal, vertical)
        self.timestep += 1
        return self.R[self.state_prev,action][0], copy.deepcopy(self.state), done

    def argmax(self,b):
        return np.random.choice(np.where(b == b.max())[0])
    

def make_MDP(epLen=2,nBottom = 24):
'''
Makes the benchmark RiverSwim MDP.
Args:
    NULL - works for default implementation
Returns:
    riverSwim - Tabular MDP environment '''
nState = nBottom + 3
nAction = 2
R_true = {}
P_true = {}
states = {}
for s in range(nState):
    states[(s)] = 0.0
    for a in range(nAction):
        R_true[s, a] = (0, 0)
        P_true[s, a] = np.zeros(nState)

# Rewards
R_true[1,0] = (-1, 0)
R_true[1,1] = (-1, 0)
R_true[2,0] = (1, 0)
R_true[2,1] = (1, 0)
n = int(nBottom/4)
#Transitions 
P_true[0,0][1] = 1.0
P_true[0,1][2] = 1.0
P_true[1,0][3:3+n] = 1/n
P_true[1,1][3+n:3+2*n] = 1/n
P_true[2,0][3+2*n:3+3*n] = 1/n
P_true[2,1][3+3*n:3+4*n] = 1/n


MDP = TabularMDP(nState, nAction, epLen)
MDP.R = R_true
MDP.P = P_true
MDP.states = states
MDP.reset()

return MDP

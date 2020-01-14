#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from tqdm import tqdm
import copy
from scipy.stats import bernoulli


# In[2]:


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


# In[76]:


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

    for s in range(nState):
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
        pContinue - 0/1 - flag for end of the episode
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

        if self.timestep == self.epLen:
            pContinue = 1
            #newState = None
            self.reset()
        else:
            pContinue = 0

        return reward, newState, pContinue
    
    def argmax(self,b):
        return np.random.choice(np.where(b == b.max())[0])


# In[102]:


class UCRL_VTR(object):
    '''
    Algorithm 1 as described in the paper Model-Based RL with
    Value-Target Regression
    '''
    def __init__(self,env,K):
        self.env = env
        self.K = K
        # Here the dimension (self.d) for the Tabular setting is |S x A x S| as stated in Appendix B
        self.d = env.nState * env.nAction * env.nState 
        # In the tabular setting the basis models is just the dxd identity matrix, see Appendix B
        self.P_basis = np.identity(self.d)
        #Our Q-values are initialized as a 2d numpy array, will eventually convert to a dictionary
        self.Q = np.zeros((env.nState,env.nAction))
        #Our State Value function is initialized as a 1d numpy error, will eventually convert to a dictionary
        self.V = np.zeros(env.nState)
        #The index of each (s,a,s') tuple, see Appendix B
        self.sigma = {}
        self.createSigma()
        #See Step 2, of algorithm 1
        self.M = pow(env.epLen,2)*self.d*np.identity(self.d)
        #See Step 2
        self.w = np.zeros(self.d)
        #See Step 2
        self.theta = np.matmul(np.linalg.inv(self.M),self.w)
        #See Step 3
        self.delta = 1/self.K
        #C_theta >= the 2-norm of theta_star, see Assumption 1
        self.C_theta = 3.0
        #A matrix that stores observed rewards that are need for the Q-updated, see equation 4
        self.r = np.zeros((env.nState,env.nAction))
        #Initialize the predicted value of the basis models, see equation 3
        self.X = np.zeros((env.epLen,self.d))
    
    def compute_Q(self,s,a,k,h):
        '''
        A function that updates both Q and V, Q is updated according to equation 4 and 
        V is updated according to equation 2
        Inputs:
            s - the state
            a - the action
            k - the current episode
            h - the current timestep
        Currently, does not properly compute the Q-values but it does seem to learn theta_star
        '''
        self.Q[s,a] = self.r[s,a] + np.dot(self.X[h,:],self.theta) + np.sqrt(self.Beta(k))             * np.sqrt(np.dot(np.dot(np.transpose(self.X[h,:]),np.linalg.inv(self.M)),self.X[h,:]))
        self.V[s] = max(self.Q[s,:])
    
    def compute_Qend(self,k,h):
        '''
        A function that updates both Q and V at the end of each episode, see step 16 of algorithm 1
        Inputs:
            k - the current episode
            h - the current timestep
        
        Currently, does not properly compute the Q-values, however, it does learn theta_star
        '''
        #step 16
        for s in range(env.nState):
            for a in range(env.nAction):
                self.Q[s,a] = self.r[s,a] + np.dot(self.X[h,:],self.theta) + np.sqrt(self.Beta(k))                     * np.sqrt(np.dot(np.dot(np.transpose(self.X[h,:]),np.linalg.inv(self.M)),self.X[h,:]))
            self.V[s] = max(self.Q[s,:])
    
    def value_vector(self,s,a,s_,h):
        '''
        A function that performs steps 9-13 of algorithm 1
        Inputs:
            s - the current state
            a - the action
            s_ - the next state
            k - the current episode
            h - the current timestep
        '''
        #Step 10
        sums = np.zeros(self.d)
        for ss in range(env.nState):
            sums += self.V[ss] * self.P_basis[self.sigma[(s,a,ss)]]
        self.X[h,:] = sums
        #Step 11
        if s_ != None:
            y = self.V[s_]
        else:
            y = 0.0
        #Step 12
        self.M = self.M + np.outer(self.X[h,:],self.X[h,:])
        #Step 13
        self.w = self.w + y * self.X[h,:]
    
    def update(self):
        '''
        Updates our approximation of theta_star at the end of each episode, see 
        Step 15 of algorithm1
        '''
        #Step 15
        self.theta = np.matmul(np.linalg.inv(self.M),self.w)
        
    def act(self,s,h):
        '''
        Returns the greedy action with respect to Q_{h,k}(s,a) for a \in A
        see step 8 of algorithm 1
        Inputs:
            s - the current state
            h - the current timestep
        '''
        #step 8
        #return env.argmax(self.Q[s,:])
        return bernoulli.rvs(0.9) #A random policy for testing
        
    def createSigma(self):
        '''
        A simple function that creates sigma according to Appendix B.
        Here sigma is a dictionary who inputs is a tuple (s,a,s') and stores
        the interger index to be used in our basis model P.
        '''
        i = 0
        for s in range(env.nState):
            for a in range(env.nAction):
                for s_ in range(env.nState):
                    self.sigma[(s,a,s_)] = int(i)
                    i += 1
    
    def Beta(self,k):
        '''
        A function that return Beta_k according to Algorithm 1, step 3
        '''
        #Step 3
        return 16*pow(self.C_theta,2)*pow(env.epLen,2)*self.d*np.log(1+env.epLen*k)             *np.log(pow(k+1,2)*env.epLen/self.delta)*np.log(pow(k+1,2)*env.epLen/self.delta)
        


# In[103]:


env = make_riverSwim(epLen = 40, nState = 4)
K = 1000
agent = UCRL_VTR(env,K)
count = np.zeros((env.nState,env.nState))
for k in tqdm(range(1,K+1)):
    env.reset()
    done = 0
    while done != 1:
        s = env.state
        h = env.timestep
        a = agent.act(s,h)
        r,s_,done = env.advance(a)
        count[s,s_] += 1
        agent.r[s,a] = r
        agent.compute_Q(s,a,k,h)
        agent.value_vector(s,a,s_,h)
    agent.update()
    agent.compute_Qend(k,h)


# In[106]:


true_p = []
for values in env.P.values():
    for value in values:
        true_p.append(value)
print('The 2-norm of (P_true - theta_star) is:',np.linalg.norm(true_p-agent.theta))


# In[105]:





# In[ ]:





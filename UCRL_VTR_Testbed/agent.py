# !/usr/bin/env python


import numpy as np
import random
import copy
from scipy.stats import bernoulli
from tqdm import tqdm

from environments import make_riverSwim, deep_sea, TabularMDP

class RLSVI(object):
    def __init__(self,env,K):
        self.env = env
        self.K = K
        self.buffer = {key: [] for key in env.R.keys()}
        self.buf = []
        self.Q = {key: 0.0 for key in self.buffer.keys()}
        self.prior_variance = 1.0
        self.noise_mean = -0.1
        self.noise_variance = 0.05
        self.action_set = (0, 1)

    def _random_argmax(self, action_values):
        argmax_list = np.where(action_values==np.max(action_values))[0]
        return self.action_set[argmax_list[np.random.randint(argmax_list.size)]]

    def name(self):
        return 'Running: RLSVI'

    def act(self,s):
        x = [self.Q[(s,a)] for a in range(self.env.nAction)]
        return self.env.argmax(np.array(x))

    def update_buffer(self,data):
        s,a,r,s_ = data[0],data[1],data[2],data[3]
        self.buffer[(s,a)].append((r,s_))

    def learn_from_buffer(self):
        perturbed_buffer = {key: [(transition[0] + np.sqrt(self.noise_variance) * np.random.randn() + self.noise_mean,
                                   transition[1]) for transition in self.buffer[key]]
                            for key in self.buffer.keys()}
        random_Q = {key: np.sqrt(self.prior_variance) * np.random.randn() for key in self.buffer.keys()}
        Q = {key: 0.0 for key in self.buffer.keys()}
        Q_temp = {key: 0.0 for key in self.buffer.keys()}
        for n in range(self.env.epLen):
            for key in self.buffer.keys():
                q = 0.0
                for transition in perturbed_buffer[key]:
                    if transition[1] == None:
                        q += transition[0]
                    else:
                        #print(Q)
                        v = max(Q[(transition[1], a)] for a in self.action_set)
                        q += transition[0] + v
                Q_temp[key] = (1.0 / ((len(self.buffer[key]) / self.noise_variance) + (1.0 / self.prior_variance))) \
                              * ((q / self.noise_variance) + (random_Q[key] / self.prior_variance))
            Q = Q_temp
            Q_temp = {key: 0.0 for key in self.buffer.keys()}
        self.Q = Q

    def run(self):
        R = []
        reward = 0.0
        for l in tqdm(range(0,self.K)):
            #if (l + 1 % 150) == 0:
            #    agent.buffer_reset()
            self.env.reset()
            done = 0
            while done != 1:
                s = self.env.state
                #print(s)
                a = self.act(s)
                r,s_,done = self.env.advance(a)
                reward += r
                #print(r)
                if done == 1:
                    self.update_buffer((s,a,r,None))
                else:
                    self.update_buffer((s,a,r,s_))

            self.learn_from_buffer()
            R.append(reward)
        return R

class PSRL(object):
    def __init__(self,env):
        self.env = env
        self.alpha = {key: np.zeros(env.nState) for key in env.R.keys()} # need a more efficent method of creating this
        self.mu = {key: 0.0 for key in env.R.keys()}
        self.sigma2 = {key: 1.0 for key in env.R.keys()} #also this can be higher, no idea how initialization of this matrix effects learning, I assume higher initial values "slows" learning
        self.sigma_r = 0.00001 #since our reward is deterministic, this keeps from the estimate of the reward from centering around the mean reward can this value be learned?
        self.buffer = {h: [] for h in range(env.epLen)}
        self.P = {key: np.zeros(env.nState) for key in env.R.keys()}
        self.R = {key: 0.0 for key in env.R.keys()}
        self.Rbar = {key: 0.0 for key in env.R.keys()}
        self.Q = {key: 0.0 for key in env.R.keys()}

    def act(self,s):
        x = np.array([self.Q[(s,a)] for a in range(self.env.nAction)])
        return env.argmax(x)

    def learn(self,l):
        self.update_statistics(l)
        self.update_priors()
        self.update_value_functions()

    def update_buffer(self,s,a,r,s_,t):
        self.buffer[t].append((s,a,r,s_))

    def update_statistics(self,l):
        for d in self.buffer.values():
            #print(d)
            s,a,r,s_ = d[l][0],d[l][1],d[l][2],d[l][3]
            if s_ != None:
                self.alpha[(s,a)][s_] = self.alpha[(s,a)][s_] + 1
            self.mu[(s,a)] = (1/self.sigma2[(s,a)]*self.mu[(s,a)] + 1/self.sigma_r*r)/(1/self.sigma2[(s,a)] + 1/self.sigma_r)
            self.sigma2[(s,a)] = 1 / (1/self.sigma2[(s,a)] + 1/self.sigma_r)

    def update_priors(self):
        for s in range(env.nState):
            for a in range(env.nAction):
                self.Rbar[(s,a)] = np.random.normal(self.mu[(s,a)],self.sigma2[(s,a)])
                self.R[(s,a)] = np.random.normal(self.Rbar[s,a],self.sigma_r)
                self.P[(s,a)] = np.random.dirichlet(self.alpha[(s,a)] + 0.003) #since numpy's dirichlet doesn't take zeros add a small numerical value for stability

    def update_value_functions(self):
        Q = {key: 0.0 for key in env.R.keys()}
        V = np.zeros(env.nState) #need to make this a dictionary
        for _ in range(env.epLen+10): #why not? Seems to improve 'stability' to convergence of 'optimal' Q-values
            for s in range(env.nState):
                for a in range(env.nAction):
                    w = np.random.normal(0,(pow(env.epLen+1,2))/(max(sum(self.alpha[s,a])-2,1)))
                    Q[(s,a)] = self.R[(s,a)] + np.inner(self.P[(s,a)],V) + w
                V[s] = max([Q[(s,a_)] for a_ in range(env.nAction)])
        self.Q = Q.copy()
        self.V = V

class UCRL_VTR(object):
    '''
    Algorithm 1 as described in the paper Model-Based RL with
    Value-Target Regression
    The algorithm assumes that the rewards are in the [0,1] interval.
    '''
    def __init__(self,env,K,random_explore,informationBonus):
        self.env = env
        self.K = K
        # A unit test that randomly explores for a period of time then learns from that experience
        # Here self.random_explore is a way to select a period of random exploration.
        # When the current episode k > total number of episodes K divided by self.random_explore
        # the algorithm switches to the greedy action with respect to its action value Q(s,a).
        if random_explore:
            self.random_explore = 10
        else:
            self.random_explore = self.K
        # For using either Bandit Books Bonus or MatrixRL bonus.
        self.informationBonus = informationBonus
        # Here the dimension (self.d) for the Tabular setting is |S x A x S| as stated in Appendix B
        self.d = self.env.nState * self.env.nAction * self.env.nState
        # In the tabular setting the basis models is just the dxd identity matrix, see Appendix B
        self.P_basis = np.identity(self.d)
        #Our Q-values are initialized as a 2d numpy array, will eventually convert to a dictionary
        self.Q = {(h,s,a): 0.0 for h in range(self.env.epLen) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        #Our State Value function is initialized as a 1d numpy error, will eventually convert to a dictionary
        self.V = {(h,s): 0.0 for s in self.env.states.keys() for h in range(env.epLen + 1)} # self.V[env.epLen] stays zero
        #self.create_value_functions()
        #The index of each (s,a,s') tuple, see Appendix B
        self.sigma = {}
        self.state_idx = {}
        self.createIdx()
        #See Step 2, of algorithm 1
#         self.M = env.epLen**2*self.d*np.identity(self.d)
        # For use in the confidence bound bonus term, see Beta function down below
        self.lam = 1.0
        #Self.L is no longer need, but will keep for now.
        self.L = 1.0
        self.M = np.identity(self.d)*self.lam
        self.Minv = np.identity(self.d)*(1/self.lam)
        #See Step 2
        self.w = np.zeros(self.d)
        #See Step 2
        self.theta = np.dot(self.Minv,self.w)
        #See Step 3
        self.delta = 1/self.K
        #m_2 >= the 2-norm of theta_star, see Bandit Algorithms Theorem 20.5
        #self.error()
        #self.m_2 = np.linalg.norm(self.true_p) + 0.1
        self.m_2 = 5.5
#         #Initialize the predicted value of the basis models, see equation 3
#         self.X = np.zeros((env.epLen,self.d))
        #See Assumptions 2,2' and Theorem 1, this equals 1 in the tabular case
        self.C_phi = 1.0
        # See Assumption 2'(Stronger Feature Regularity), and consider the case when v_1 = v_2 = ....
        self.C_psi = np.sqrt(env.nState)
        # See Theorem 1
        self.C_M = 1.0
        # See Theorem 1
        self.C_psi_ = 1.0
        # This value scales our confidence interval, must be > 0
        self.c = 0.1
        self.d1 = env.nState * env.nAction




    def feature_vector(self,s,a,h):
        '''
        Returning sum_{s'} V[h+1][s'] P_dot(s'|s,a),
        with V stored in self.
        Inputs:
            s - the state
            a - the action
            h - the current timestep within the episode
        '''
        sums = np.zeros(self.d)
        for s_ in self.env.states.keys():
            #print(s,s_)
            sums += self.V[h+1,s_] * self.P_basis[self.sigma[(s,a,s_)]]
        return sums

    def proj(self, x, lo, hi):
        '''Projects the value of x into the [lo,hi] interval'''
        return max(min(x,hi),lo)

    def update_Q(self,s,a,k,h):
        '''
        A function that updates both Q and V, Q is updated according to equation 4 and
        V is updated according to equation 2
        Inputs:
            s - the state
            a - the action
            k - the current episode
            h - the current timestep within the episode
        Currently, does not properly compute the Q-values but it does seem to learn theta_star
        '''
        #Here env.R[(s,a)][0] is the true reward from the environment
        # Alex's code: X = self.X[h,:]
        # Suggested code:
        X = self.feature_vector(s,a,h)
        self.Q[h,s,a] = self.proj(self.env.R[(s,a)][0] + np.dot(X,self.theta) + self.Beta(k) \
            * np.sqrt(np.dot(np.dot(np.transpose(X),self.Minv),X)), 0, self.env.epLen + self.informationBonus) #Does this value have to change???
        self.V[h,s] = max(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))

    def update_Qend(self,k):
        '''
        A function that updates both Q and V at the end of each episode, see step 16 of algorithm 1
        Inputs:
            k - the current episode
        '''
        #step 16
        for h in range(self.env.epLen-1,-1,-1):
            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                    #Here env.R[(s,a)][0] is the true reward from the environment
                    # Alex's code: X = self.X[h,:]
                    # Suggested code:
                    self.update_Q(s,a,k,h)
                self.V[h,s] = max(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))

    def update_stat(self,s,a,s_,h):
        '''
        A function that performs steps 9-13 of algorithm 1
        Inputs:
            s - the current state
            a - the action
            s_ - the next state
            k - the current episode
            h - the timestep within episode when s was visited (starting at zero)
        '''
        #Step 10
#         self.X[h,:] = self.feature_vector(s,a,h) # do not need to store this
        X = self.feature_vector(s,a,h)
        #Step 11
        y = self.V[h+1,s_]
#         if s_ != None:
#             y = self.V[h+1][s_]
#         else:
#             y = 0.0
        #Step 12
        self.M = self.M + np.outer(X,X)
        self.Minv = self.Minv - np.dot((np.outer(np.dot(self.Minv,X),X)),self.Minv) / \
                    (1 + np.dot(np.dot(X,self.Minv),X))
        #Step 13
        self.w = self.w + y*X

    def update_param(self):
        '''
        Updates our approximation of theta_star at the end of each episode, see
        Step 15 of algorithm1
        '''
        #Step 15
        #print(self.M)
        self.theta = np.matmul(self.Minv,self.w)

    def act(self,s,h,k):
        '''
        Returns the greedy action with respect to Q_{h,k}(s,a) for a \in A
        see step 8 of algorithm 1
        Inputs:
            s - the current state
            h - the current timestep within the episode
        '''
        #step 8
        if k > (self.K / self.random_explore - 1):
            #print (max(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)])))
            return self.env.argmax(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))
        else:
            return bernoulli.rvs(0.9) #A random policy for testing

    def createIdx(self):
        '''
        A simple function that creates sigma according to Appendix B.
        Here sigma is a dictionary who inputs is a tuple (s,a,s') and stores
        the interger index to be used in our basis model P.
        '''
        i = 0
        j = 0
        k = 0
        for s in self.env.states.keys():
            self.state_idx[s] = int(j)
            j += 1
            for a in range(self.env.nAction):
                for s_ in self.env.states.keys():
                    self.sigma[(s,a,s_)] = int(i)
                    i += 1

    def Beta(self,k):
        '''
        A function that return Beta_k according to Algorithm 1, step 3
        '''
        #Step 3
        #Bonus as according to step 3
        #return 16*pow(self.m_2,2)*pow(env.epLen,2)*self.d*np.log(1+env.epLen*k) \
        #    *np.log(pow(k+1,2)*env.epLen/self.delta)*np.log(pow(k+1,2)*env.epLen/self.delta)

        #Confidence bound from Chapter 20 of the Bandit Algorithms book, see Theorem 20.5.
        first = np.sqrt(self.lam)*self.m_2
        (sign, logdet) = np.linalg.slogdet(self.M)
        #second = np.sqrt(2*np.log(1/self.delta) + self.d*np.log((self.d*self.lam + k*self.L*self.L)/(self.d*self.lam)))
        det = sign * logdet
        second = np.sqrt(2*np.log(1/self.delta) + np.log(k) + min(det,pow(10,10)) - np.log(pow(self.lam,self.d)))
        return first + second
        '''
        elif self.select_bonus == 2:
            first = self.c*(self.C_M * self.C_psi_ ** 2)
            second = np.log(self.K*self.env.epLen*self.C_phi)*self.d1
            #The line of code below for an anytime version
            #second = np.log(k*self.env.epLen*self.C_phi)*self.d1
            return np.sqrt(first*second)
        else:
            print('self.select_bonus must be either 1 or 2')
        '''



    def run(self):
        R = []
        reward = 0.0
        for k in tqdm(range(1,self.K+1)):
            self.env.reset()
            done = 0
            while done != 1:
                s = self.env.state
                h = self.env.timestep
                a = self.act(s,h,k)
                r,s_,done = self.env.advance(a)
                reward += r
                self.update_stat(s,a,s_,h)
            self.update_param()
            self.update_Qend(k)
            R.append(reward)
        return R

    def name(self):
        return 'Running: UCRL_VTR'

    #def error(self):
    #    self.true_p = []
    #    for values in self.env.P.values():
    #        for value in values:
    #            self.true_p.append(value)
    #    print('The 2-norm of (P_true - theta_star) is:',np.linalg.norm(self.true_p-self.theta))

class UCBVI(object):
    def __init__(self,env,K):
        self.env = env
        self.K = K
        self.delta = 1/self.K
        self.buffer = {h: [] for h in range(self.env.epLen)}
        self.Nxay = {(s,a,s_): 0.0 for s in self.env.states.keys() for a in range(self.env.nAction) \
                     for s_ in self.env.states.keys()}
        self.Nxa = {(s,a): 0.0 for s in self.env.states.keys() for a in range(self.env.nAction)}
        self.N_ = {(h,s,a): 0.0 for h in range(self.env.epLen+1) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        self.P = {(s,a,s_): 0.0 for s in self.env.states.keys() for a in \
                  range(self.env.nAction) for s_ in self.env.states.keys()}
        self.Q = {(h,s,a): self.env.epLen+1 for h in range(self.env.epLen) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}


    def update_buffer(self,s,a,r,s_,h):
        self.buffer[h].append((s,a,r,s_,h))

    def act(self,s,h):
        x = np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)])
        return self.env.argmax(x)

    def learn(self,k):
        self.update_counts(k)
        self.update_probability_transition()
        self.update_value_functions(k)

    def update_probability_transition(self):
        for s in self.env.states.keys():
            for a in range(self.env.nAction):
                if self.Nxa[(s,a)] > 0:
                    for s_ in self.env.states.keys():
                        self.P[(s,a,s_)] = (self.Nxay[(s,a,s_)]) / (self.Nxa[(s,a)])

    def update_counts(self,k):
        for d in self.buffer.values():
            #print(d)
            s,a,r,s_,h = d[k][0],d[k][1],d[k][2],d[k][3],d[k][4]
            if s_ != None:
                self.Nxay[(s,a,s_)] += 1
            self.Nxa[(s,a)] += 1
            self.N_[(h,s,a)] += 1

    def update_value_functions(self,k):
        V = {(h,s): 0.0 for s in self.env.states.keys() for h in range(env.epLen + 1)}
        for h in range(self.env.epLen-1,-1,-1):
            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                    if self.Nxa[(s,a)] > 0:
                        #bonus = self.bonus_1(s,a)
                        PV = self.multiplyDictionaries(s,a,h,V)
                        bonus = self.bonus_2(s,a,h,V)
                        self.Q[(h,s,a)] = min(min(self.Q[(h,s,a)], self.env.epLen), self.env.R[(s,a)][0] + PV + bonus)
                    else:
                        self.Q[(h,s,a)] = self.env.epLen
                V[(h,s)] = max(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))


    def bonus_1(self,s,a):
        T = self.K * self.env.epLen
        L = np.log(5 * self.env.nState * self.env.nAction  * T / self.delta)
        return 7 * env.epLen * L * np.sqrt(1 / self.Nxa[(s,a)])

    def bonus_2(self,s,a,h,V):
        temp = []
        sums = 0.0
        T = self.K * self.env.epLen
        L = np.log(5 * self.env.nState * self.env.nAction  * T / self.delta)
        for s_ in self.env.states.keys():
            c = V[(h+1,s_)] * self.P[(s,a,s_)]
            temp.append(c)
            min1 = pow(100,2)*pow(self.env.epLen,3) * pow(self.env.nState,2)*self.env.nAction*L*L \
                /max(self.N_[(h+1,s_,a)],0.001)
            sums += self.P[(s,a,s_)] * min(min1,self.env.epLen)
        var = np.var(temp)
        #print(var)
        first = np.sqrt(8*L*var/self.Nxa[(s,a)])
        second = 14*self.env.epLen*L/(3*self.Nxa[(s,a)])
        third = np.sqrt(8*sums/self.Nxa[(s,a)])
        return first + second + third

    def multiplyDictionaries(self,s,a,h,V):
        sums = 0.0
        for s_ in self.env.states.keys():
            sums += V[(h+1,s_)] * self.P[(s,a,s_)]
        return sums

    def name(self):
        return 'Running: UCBVI'

    def run(self):
        R = []
        reward = 0.0
        for k in tqdm(range(K)):
            env.reset()
            done = 0
            while done != 1:
                s = env.state
                h = env.timestep
                a = agent.act(s,h)
                r,s_,done = env.advance(a)
                reward += r
                if done != 1:
                    agent.update_buffer(s,a,r,s_,h)
                else:
                    agent.update_buffer(s,a,r,s_,h)
            agent.learn(k)
            R.append(reward)
        return R

class LSVI_UCB(object):
    def __init__(self,env,K,c):
        self.env = env
        self.K = K
        self.d = self.env.nState * self.env.nAction
        self.c = c
        self.p = 10/self.K
        self.lam = 1.0
        self.Lam = [(self.lam*np.identity(self.d)) for h in range(env.epLen)]
        self.phi = np.identity(self.d)
        self.Q = {(h,s,a): 0.0 for h in range(self.env.epLen+1) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        self.w = [np.zeros(self.d) for h in range(env.epLen)]
        self.wsum = [np.zeros(self.d) for h in range(env.epLen)]
        self.buffer = {h: [] for h in range(self.env.epLen)}
        self.sigma = {}
        self.createSigma()

    def learn(self,s,a,r,s_,h):
        self.Lam[h] += np.outer(self.phi[self.sigma[s,a]],self.phi[self.sigma[s,a]])
        self.wsum[h] += self.phi[self.sigma[s,a]] * (r + \
                                max(np.array([self.Q[(h+1,s_,a_)] for a_ in range(self.env.nAction)])))
        self.w[h] = np.matmul(np.linalg.inv(self.Lam[h]),self.wsum[h])


    def update_Q(self,k):
        for h in range(self.env.epLen-1,-1,-1):
            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                    min1 = np.dot(self.w[h],self.phi[self.sigma[s,a]]) + self.Beta(h,k)* \
                                  np.sqrt(np.dot(np.dot(self.phi[self.sigma[s,a]].T,np.linalg.inv(self.Lam[h])) \
                                                  ,self.phi[self.sigma[s,a]]))
                    min2 = self.env.epLen
                    self.Q[h,s,a] = self.proj(min1,0,self.env.epLen)

    def Beta(self,h,k):

        first = np.sqrt(self.lam)*self.c
        #second = np.sqrt(2*np.log(1/self.delta) + self.d*np.log((self.d*self.lam + k*self.L*self.L)/(self.d*self.lam)))
        (sign, logdet) = np.linalg.slogdet(self.Lam[h])
        det = sign * logdet
        second = np.sqrt(2*np.log(1/self.p) + np.log(k) + min(det,pow(10,10)) - np.log(pow(self.lam,self.d)))
        return first + second


        #itoa = np.log(2*self.d*self.K*self.env.epLen/self.p)
        #return self.c*self.d*self.env.epLen*np.sqrt(itoa)

    def createSigma(self):
        '''
        A simple function that creates sigma according to Appendix B.
        Here sigma is a dictionary who inputs is a tuple (s,a,s') and stores
        the interger index to be used in our basis model P.
        '''
        i = 0
        for s in self.env.states.keys():
            for a in range(self.env.nAction):
                self.sigma[(s,a)] = int(i)
                i += 1

    def act(self,s,h):
        x = np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)])
        return self.env.argmax(x)

    def proj(self,x, lo, hi):
        return max(min(x,hi),lo)

    def name(self):
        return 'Running: LSVI-UCB'

    def run(self):
        R = 0
        Rvec = []
        for k in tqdm(range(1,self.K+1)):
            self.env.reset()
            done = 0
            while done != 1:
                s = self.env.state
                h = self.env.timestep
                a = self.act(s,h)
                r,s_,done = self.env.advance(a)
                R+=r
                self.learn(s,a,r,s_,h)
            Rvec.append(R)
            self.update_Q(k)
        return Rvec

class Optimal_Agent(object):
    def __init__(self,env,a,K):
        self.env = env
        self.a = a
        self.K = K

    def name(self):
        return 'Running: Optimal_Policy'

    def run(self):
        R = 0
        Rvec = []
        for k in tqdm(range(1,self.K+1)):
            self.env.reset()
            done = 0
            while done != 1:
                s = self.env.state
                h = self.env.timestep
                r,s_,done = self.env.advance(self.a)
                R+=r
            Rvec.append(R)
        return Rvec

class UC_MatrixRL(object):
    def __init__(self,env,N):
        self.env = env
        #See Step 3 of Algorithm 1
        self.N = N
        #The dimensionality of phi(s,a), see Step 2 or Assumption 1
        self.d1 = self.env.nState * self.env.nAction
        #The dimensionality of psi(s'), see Step 2 or Assumption 1
        self.d2 = self.env.nState
        #Step 2
        self.features_state_action = {(s,a): np.zeros(self.d1) for s in self.env.states.keys() for a in range(self.env.nAction)}
        #Step 3
        self.features_next_state = {(s): np.zeros(self.d2) for s in self.env.states.keys()}
        # A hack for using numpy's linear algebra functions in Step 4
        self.features_next_state_mat = np.identity(self.d2)
        # Creates the Identity Matrix for a dictionary
        self.createIdentity()
        # Initialize our Q matrix
        self.Q = {(h,s,a): 0.0 for h in range(self.env.epLen) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        #Step 4, this step maybe unnecessary since we are using Sherman-Morrison
        self.A = np.identity(self.d1)
        #For use in the Sherman-Morrison Update
        self.Ainv = np.linalg.inv(self.A)
        #Step 4
        self.M = np.zeros((self.d1,self.d2))
        #See Assumptions 2,2' and Theorem 1, this equals 1 in the tabular case
        self.C_phi = 1.0
        # See Assumption 2'(Stronger Feature Regularity), and consider the case when v_1 = v_2 = ....
        self.C_psi = np.sqrt(env.nState)
        # See Theorem 1
        self.C_M = 1.0
        # See Theorem 1
        self.C_psi_ = 1.0
        # This value scales our confidence interval, must be > 0
        self.c = 1.0
        # For use in updating M_n, see Step 13 and Eqn (2)
        self.sums = np.zeros((self.d1,self.d2))
        #Creates K_psi, see Section 3.1: Estimating the core matrix.
        self.createK()
        self.lam = 1.0
        self.delta = 1/self.N

    def createK(self):
        '''
        A function that creates K_psi and (K_psi)^-1 in computing M_n
        See Section 3.1: Estimating the core matrix
        '''
        self.K = np.zeros((self.d2,self.d2))
        for s_ in self.env.states.keys():
            self.K = self.K + np.outer(self.features_next_state[s_],self.features_next_state[s_])
        self.Kinv = np.linalg.inv(self.K)

    def act(self,s,h):
        '''
        A function that returns the argmax of Q given the state and timestep
        '''
        return self.env.argmax(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))

    def createIdentity(self):
        '''
            A function that creates the Identity Matrix for a Dictionary
        '''
        i = 0
        for key in self.features_state_action.keys():
            self.features_state_action[key][i] = 1
            i += 1
        j = 0
        for key in self.features_next_state.keys():
            self.features_next_state[key][j] = 1
            j += 1

    def proj(self, x, lo, hi):
        '''Projects the value of x into the [lo,hi] interval'''
        return max(min(x,hi),lo)

    def compute_Q(self,n):
        '''
        A function that computes the Optimisic Q-Values, see step 6 and Equations 4,8.
        '''
        Q = {(h,s,a): 0.0 for h in range(self.env.epLen) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        V = {h: np.zeros(self.env.nState) for h in range(self.env.epLen + 1)}
        for h in range(self.env.epLen-1,-1,-1):
            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                    #For use in computing Q, like in UCRL_VTR we have access to the true reward function.
                    r = self.env.R[s,a][0]

                    value = np.dot(np.matmul(np.dot(self.features_state_action[s,a].T,self.M),\
                            self.features_next_state_mat),V[h+1])

                    bonus = 2 * self.C_psi * h * self.Beta(n) * np.dot(\
                           np.dot(self.features_state_action[s,a],self.Ainv),self.features_state_action[s,a])
                    # Computing the optimistic Q-values as according to Eqn (8).
                    Q[h,s,a] = self.proj(r+value+bonus,0,self.env.epLen)
                V[h][s] = max(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))
        self.Q = Q.copy()

    '''
    def Beta_2(self,n):

        A function that computes Beta under the assumption that Theorem 2 holds, see equation 8
        with the multiplicitive H term included this is no longer an optimal bonus.

        first = self.c*(self.C_M * self.C_psi_ ** 2)
        second = np.log(self.N*self.env.epLen*self.C_phi)*self.d1
        #The line of code below for an 'anytime' version - there are no theoretical garanuntees for this statement.
        #second = np.log(n*self.env.epLen*self.C_phi)*self.d1
        return first * second
    '''

    def Beta(self,n):
        '''
        A function that return Beta_k according to Theorem 20.5 in Bandits book
        Also, if you return np.sqrt(first + second) instead of first + second then
        you get higher cumlative reward. However, in Theorem 19.2, Beta_n is already
        under the square root.
        '''
        #Step 3
        #Bonus as according to step 3
        '''
        return np.sqrt(16*pow(self.m_2,2)*pow(env.epLen,2)*self.d*np.log(1+env.epLen*k) \
            *np.log(pow(k+1,2)*env.epLen/self.delta)*np.log(pow(k+1,2)*env.epLen/self.delta))
        '''

        #Confidence bound from Chapter 20 of the Bandit Algorithms book, see Theorem 20.5.
        first = np.sqrt(self.lam)*np.sqrt(self.C_M*self.d1)
        (sign, logdet) = np.linalg.slogdet(scipy.linalg.sqrtm(self.A))
        #second = np.sqrt(2*np.log(1/self.delta) + self.d*np.log((self.d*self.lam + k*self.L*self.L)/(self.d*self.lam)))
        det = sign * logdet
        second = np.sqrt(2*np.log(1/self.delta) + np.log(n) + min(det,pow(10,10)) - np.log(pow(self.lam,self.d1)))
        #print(det)
        return first + second

    def update_core_matrix(self,s,a,s_):
        '''
        A function that performs step 12 and 13.
        '''
        #While the line below is in Algorithm 1, when using the Sherman Morrison update it is not needed.
        #self.A = self.A + np.outer(self.features_state_action[s,a],self.features_state_action[s,a])
        #Sherman Morrison Update (Rich's book Eqn 9.22)
        self.Ainv = self.Ainv - np.dot((np.outer(np.dot(self.Ainv,self.features_state_action[s,a]) \
                 ,self.features_state_action[s,a])),self.Ainv) / \
                    (1 + np.dot(np.dot(self.features_state_action[s,a],self.Ainv),self.features_state_action[s,a]))
        #The summation step of Eqn (2)
        self.sums = self.sums + np.outer(self.features_state_action[s,a],self.features_next_state[s_])
        #Here (K_psi)^-1 was pre-computed during the initialization. See Eqn (2)
        self.M = np.matmul(np.matmul(self.Ainv,self.sums),self.Kinv)

    def name(self):
        return 'Running: UC_MatrixRL'

    def run(self):
        R = 0
        Rvec = []
        for n in tqdm(range(1,self.N+1)):
            self.env.reset()
            done = 0
            self.compute_Q(n)
            while not done:
                s = self.env.state
                h = self.env.timestep
                a = self.act(s,h)
                r,s_,done = self.env.advance(a)
                R += r
                self.update_core_matrix(s,a,s_)
            Rvec.append(R)
        return Rvec

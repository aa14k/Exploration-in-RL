import numpy as np
from tqdm import tqdm
from envs import mountaincar
import tiles3 as tc
import logging

class MountainCarTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
    
    def get_tiles(self, position, velocity):
        """
        Takes in a position and velocity from the mountaincar environment
        and returns a numpy array of active tiles.
        
        Arguments:
        position -- float, the position of the agent between -1.2 and 0.5
        velocity -- float, the velocity of the agent between -0.07 and 0.07
        returns:
        tiles - np.array, active tiles
        """

        POSITION_MIN = -1.2
        POSITION_MAX = 0.5
        VELOCITY_MIN = -0.07
        VELOCITY_MAX = 0.07

        position_scale = self.num_tiles / (POSITION_MAX - POSITION_MIN)
        velocity_scale = self.num_tiles / (VELOCITY_MAX - VELOCITY_MIN)

        tiles = tc.tiles(self.iht, self.num_tilings, [position * position_scale, 
                                                      velocity * velocity_scale])
        
        return np.array(tiles)

class LinearDyna(object):
    def __init__(self,env,K,tilings,tiles,iht_size,steps,alpha_l,alpha_p,tau,gamma,B):
        self.env = env
        self.K = K
        self.num_tilings = tilings
        self.num_tiles = tiles
        self.iht_size = iht_size
        self.steps = steps
        self.alpha_l = alpha_l
        self.alpha_p = alpha_p
        self.tau = tau
        self.gamma = gamma
        self.B = B
        self.eps = 0.0
        self.num_actions = 3
        #self.theta = np.random.normal(0.0,0.0001,size=self.iht_size)
        self.theta = np.zeros(self.iht_size)
        self.F = np.zeros((self.num_actions,self.iht_size,self.iht_size))
        self.f = np.zeros((self.num_actions,self.iht_size))
        self.Phi = np.zeros((self.num_actions,self.iht_size,self.iht_size))
        self.PhiPhi_ = np.zeros((self.num_actions,self.iht_size,self.iht_size))
        for a in range(self.num_actions):
            self.Phi[a] = 0.01*np.identity(self.iht_size)
        self.I = np.identity(self.iht_size)
        self.II = self.I
        self.II[0,0] = 0.0
        self.tc = MountainCarTileCoder(iht_size=self.iht_size, 
                                         num_tilings=self.num_tilings, 
                                         num_tiles=self.num_tiles)
        self.buffer = []
        
    def mc_reward(self,position):
        '''
        The true reward function for mountain car. This is given to the agent when possible
        '''
        if position >= 0.5:
            return 0.0
        else:
            return -1.0

    def get_phi(self,tiles):
        '''
        Computes the tile coded features of a given state. Input is the active tiles output is a vector of
        size iht_size.
        '''
        #Gets the tile coded features
        phi = np.zeros(self.iht_size)
        for tile in tiles:
            phi += self.I[tile]
        return phi
    
    def act(self,s):
        
        #computes the featurized state phi
        position,velocity = s
        active_tiles = self.tc.get_tiles(position,velocity)
        phi = self.get_phi(active_tiles)
        
        #computes the Q values for each action a given featurized state phi
        b = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            b[a] = self.mc_reward(position) + self.gamma * np.dot(np.dot(self.theta,self.F[a]),phi)
        
        #Finds the argmax of the previously computed Q values, breaking ties randomly.
        a = np.random.choice(np.where(b == b.max())[0])
        return a
    
    #Commented code below is for computing the exact Least Squares Update. Currently does not work.
      
    def update(self,s,a,r,s_,done):
        '''
        Updates the values of theta, our estimated transition model F, and our estimated reward model f.
        '''
        
        
        #Computes the tile coded feature for current state s
        pos,vel = s
        active_tiles = self.tc.get_tiles(pos,vel)
        self.phi = self.get_phi(active_tiles)
        
        #Computes the tile coded feature for next state s_
        position_,velocity_ = s_
        active_tiles_ = self.tc.get_tiles(position_,velocity_)
        self.phi_ = self.get_phi(active_tiles_)
        
        #Updates our theta values using gradient descent
        self.theta = self.theta + self.alpha_l*(r + self.gamma * np.inner(self.phi_,self.theta) \
                                             - np.inner(self.phi,self.theta))*self.phi
        
        #Update our transition model using gradient descent
        #self.F[a] = self.F[a] + self.B * np.outer((self.phi_ - np.dot(self.F[a],self.phi)),self.phi)
        #Update our reward model using gradient descent
        self.Phi[a] = self.Phi[a] + np.outer(self.phi,self.phi)
        self.PhiPhi_[a] = self.PhiPhi_[a] + np.outer(self.phi,self.phi_)
        theta_outer = np.outer(self.theta,self.theta)
        I = np.identity(self.iht_size)
        temp = theta_outer + 0.999*I
        theta_inv = np.linalg.inv(temp)
        first = np.matmul(np.linalg.inv(self.Phi[a]),self.PhiPhi_[a])
        second = np.matmul(first,theta_outer)
        final = np.matmul(second,theta_inv)
        self.F[a] = final 
        self.f[a] = self.f[a] + self.B * (r - np.inner(self.f[a],self.phi)) * self.phi
        
        #Commented code below is for computing the exact solution to the OLS. Currently does not work.
        '''
        self.Dinv[a] = self.Sherman_Morrison(self.phi,self.Dinv[a])

        self.x = np.dot(self.Dinv[a],self.phi)

        #self.F[a] = self.F[a] + np.outer((self.phi_ - np.dot(self.F[a],self.phi))/(1 + np.inner(self.phi,self.x)),self.x)

        self.update_F(a)

        #self.f[a] = self.f[a] + (r - np.inner(self.phi,self.f[a]))/(1 + np.inner(self.phi,self.x))*self.x

        self.update_f(a,r)
        '''
        
        #Runs our planning step.
        self.plan()
    
    def plan(self):
        '''
        Using Dyna-style planning to update our theta estimate with simulated experience on a learnt model.
        '''
        
        #initializes the theta using in planning to be the current theta estimate
        theta_tilde = self.theta
        
        #we do the planning portion p many times
        for p in range(self.tau):
            
            #Below are different ways to sample a state s for planning
            
            #Here we sample s uniformly from the space of all states
            
            #position = np.random.uniform(-1.2,0.6)
            #velocity = np.random.uniform(-0.07,0.07)
            #active_tiles_tilde = self.tc.get_tiles(position,velocity)
            #phi_tilde = self.get_phi(active_tiles_tilde)
            
            #Here we sample s from a buffer the stores all observed states
            
            # row = np.random.randint(len(self.buffer))
            # tup = self.buffer[row]
            # tiles = self.tc.get_tiles(tup[0],tup[1])
            # phi_tilde = self.get_phi(tiles)
            
            #Here we sample s from the support. Meaning we sample a unit vector as the state
            
            row = np.random.randint(self.iht_size)
            phi_tilde = self.I[row]
            
            #compute the Q values for given our previously sampled state
            
            b = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                b[a] = np.inner(phi_tilde,self.f[a]) + self.gamma * np.dot(np.dot(theta_tilde,self.F[a]),phi_tilde)
            
            #Take the action that maximizes our q values we previously computed
            a_tilde = np.random.choice(np.where(b == b.max())[0])
            
            #Compute the featurized next state given a featurized state and non featurized action
            phi_tilde_ = np.dot(self.F[a_tilde],phi_tilde)
            #compute the reward given a featurized state and a non featurized action
            r_tilde = np.inner(phi_tilde,self.f[a_tilde])
            #Update theta using the simulated experience 
            theta_tilde = theta_tilde + self.alpha_p * (r_tilde + self.gamma * np.inner(theta_tilde,phi_tilde_) \
                                                     - np.inner(theta_tilde,phi_tilde))*phi_tilde
        #Update the current estimate of theta to be the estimate from the simulation
        self.theta = theta_tilde
     
    def update_state_buffer(self,s):
        '''
        Updates the buffer with the curretn state s
        '''
        self.buffer.append(s)
    
    def run(self):
        '''
        Runs the rl algorithm and returns the observed number of steps.
        '''
        print("Linear-Dyna")
        R = 0
        Rvec = []
        logging.basicConfig(filename='reward_999_unit.log', level=logging.INFO)
        for k in tqdm(range(1,self.K+1)):
            self.env.reset()
            done = None
            step = 0
            while not done:
                step += 1
                #if step % 50 == 0:
                #    print(step)
                s = self.env.current_state
                self.update_state_buffer(s)
                a = self.act(s)
                r,s_,done = self.env.advance(a)
                #if r > -1:
                #    print(r,done)
                if step == self.steps:
                    done = True
                self.update(s,a,r,s_,done)
            Rvec.append(step)
            logging.info(step)
        return Rvec

#number of episodes
K = 30
#num of runs
runs = 10
#the environment
env = mountaincar.mountain_car()
#number of tiles, chosen according to hengshaui's work
tiles = 8
#number of tilings, chosen according to hengshaui's work
tilings = 10
#size of the index hash table, chosen according to hengshaui's work
iht_size= 1000
#max number of interactions with an environment before a reset, chosen according to hengshaui's work
steps = 1000
#learning rate for theta
alpha_l = 0.05
#learning rate for theta tilde, should somehow scale with tau, the number of planning steps
alpha_p = 0.01
#number of planning steps
tau = 5
#The discounting factor, chosen according to hengshaui's work
gamma = 1.0
#The learning rate for updating the learnt models F and f, chosen according to hengshaui's work
B = 0.01
#A matrix the stores the step for each episode within a run.
step = np.zeros((runs,K))
for i in tqdm(range(runs)):
    agent = LinearDyna(env,K,tilings,tiles,iht_size,steps,alpha_l,alpha_p,tau,gamma,B)
    step[i,:] = agent.run()
#averages the result for each episode by the steps per run.
results = np.mean(step,axis=0)
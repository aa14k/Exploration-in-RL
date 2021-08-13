import numpy as np

class mountain_car(object):
    '''General RL environment'''

    def __init__(self):
        self.reward = 0.0
        self.observation = None
        self.termination = None
        self.current_state= None
        self.position = None
        self.velocity = None
        self.count = 0.0

    def reset(self):
        self.position = np.random.uniform(-0.6,-0.4)
        self.velocity = 0.0
        self.current_state = np.array([self.position,self.velocity])
        

    def advance(self, action):
        self.position, self.velocity = self.current_state
        
        self.termination = False
        self.reward = -1.0
        self.velocity = self.bound_velocity(self.velocity + 0.001 * (action - 1) - 0.0025 * np.cos(3 * self.position))
        self.position = self.bound_position(self.position + self.velocity)
        
        if self.position == -1.2:
            self.velocity = 0.0
        elif self.position >= 0.5:
            self.current_state = None
            self.termination = True
            self.reward = 0.0
        
        self.current_state = np.array([self.position,self.velocity])
        
        return (self.reward,self.current_state,self.termination)
    
    def bound_velocity(self, velocity):
        if velocity > 0.07:
            return 0.07
        if velocity < -0.07:
            return -0.07
        return velocity

    def bound_position(self, position):
        if position > 0.5:
            return 0.5
        if position < -1.2:
            return -1.2
        return position
    
    def argmax(self,b):
        return np.random.choice(np.where(b == b.max())[0])

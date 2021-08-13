# 98 state Boyan chain environment
# Created by Taapas Agrawal
# 28-06-2021

import numpy as np
# Constants
RIGHT = 0
SKIP = 1

class Boyan():
    def __init__(self):
        self.states = 97
        self.state = 0

    def start(self):
        self.state = 97
        return self.state

    def step(self, a):
        reward = -3
        terminal = False

        if a == SKIP and self.state<=2:
            print("Skip action is not available in state 2 or state 1... Exiting now.")
            exit()
            
        if a == RIGHT:
            if self.state ==2:
                reward = -2
            if self.state ==1:
                reward = 0
            self.state = self.state - 1
        elif a == SKIP:
            self.state = self.state - 2

        if (self.state == 0):
            terminal = True

        return (reward, self.state, terminal)
    
    def getPR(self):

        P = np.zeros((98, 98))
        for i in range(97, 0):
            P[i, i-1] = .5
            P[i, i-2] = .5

        P[2, 1] = 1
        P[1, 0] = 1

        R = np.array([-3]*95  + [-2, 0, 0])

        return P, R


def compute_rep_map(dim):
    l = [1] + (dim-1)*[0]
    arr = np.array([])
    arr = np.hstack((arr, np.array(l)))

    n = len(l)-1
    for i in range(n):
        j = i + 1
        while(l[i]!=0):
            l[i] = l[i] - 0.25
            l[j] = l[j] + 0.25
            arr = np.vstack((arr, np.array(l)))
    k = dim*[0]
    arr = np.vstack((arr, np.array(k)))
    return arr

class BoyanRep:
    def __init__(self):
        self.dim = 25
        self.map = compute_rep_map(self.dim)

    def getmap(self):
        return self.map

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.dim

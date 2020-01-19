# !/usr/bin/env python

import numpy as np
import random
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

from environments import make_riverSwim, deep_sea, TabularMDP
from agent import RLSVI, UCRL_VTR, UCBVI, PSRL, LSVI_UCB, Optimal_Agent, UC_MatrixRL

K = 500
epLen = 40
nState = 10

'''
agent = LSVI_UCB(make_riverSwim(epLen, nState),K,0.1) # c=0.1 seems to work, not too sure on how to set this at the moment
print(agent.name())
R_LSVI_UCB = agent.run()
#plt.plot(R_LSVI_UCB, label = 'LSVI_UCB')
'''

agent = Optimal_Agent(make_riverSwim(epLen, nState),1,K)
print(agent.name())
R_Optimal = agent.run()
plt.plot(R_Optimal, label = 'Optimal Policy')


agent = UC_MatrixRL(make_riverSwim(epLen,nState),K)
print(agent.name())
R_UC_MatrixRL = agent.run()
plt.plot(R_UC_MatrixRL, label = 'UC_MatrixRL')

'''
random_explore = False
agent = UCRL_VTR(deep_sea(num_steps = 4),K,random_explore)
print(agent.name())
R_UCRL_VTR = agent.run()
plt.plot(R_UCRL_VTR, label = 'UCRL_VTR')
'''

agent = RLSVI(make_riverSwim(epLen,nState),K)
print (agent.name())
R_RLSVI = agent.run()
plt.plot(R_RLSVI, label = 'RLSVI')

#plt.loglog(range(K), label = 'plot of n')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Cumlative Reward')
plt.title('Plot of Cumlative Reward vs Episode')
plt.grid()
plt.show()

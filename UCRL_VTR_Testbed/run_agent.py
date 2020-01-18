# !/usr/bin/env python

import numpy as np
import random
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

from environments import make_riverSwim, deep_sea, TabularMDP
from agent import RLSVI, UCRL_VTR, UCBVI, PSRL

K = 600
agent = UCRL_VTR(make_riverSwim(epLen = 20, nState = 4),K,1.0) # c=0.1 seems to work, not too sure on how to set this at the moment
print(agent.name())
R_LSVI_UCB = agent.run()
agent = RLSVI(make_riverSwim(epLen = 20, nState = 4),K)
print (agent.name())
R_RLSVI = agent.run()
random_explore = False
agent = UCRL_VTR(make_riverSwim(epLen = 20, nState = 4),K,random_explore)
print(agent.name())
R_UCRL_VTR = agent.run()

plt.plot(R_RLSVI, label = 'RLSVI')
plt.plot(R_UCRL_VTR, label = 'UCRL_VTR')
plt.plot(R_LSVI_UCB, label = 'LSVI_UCB')
#plt.loglog(range(K), label = 'plot of n')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Cumlative Reward')
plt.title('Plot of Cumlative Reward vs Episode')
plt.grid()
plt.show()

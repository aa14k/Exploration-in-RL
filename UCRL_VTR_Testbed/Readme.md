# Readme

run_agent.py - this file runs LSVI_UCB, RLSVI and UCRL_VTR in the RiverSwim and RLSVI and UCRL_VTR DeepSea environments. 
It outputs a plot of the cumlative reward of both RLSVI and UCRL_VTR. As of right now, UCRL_VTR has a hard
time running DeepSea because the dimension of the problem is |S * A * S| in the tabular setting. If depth of the problem 
is equal to 10, then we are inverting and multiplying a 20000-dimensional square matrix at each time-step. Need to either
reduce the dimension of the problem by moving the problem from the tabular to the linear setting, or try to implement more 
efficent linear algebra operations. Currently, UCRL_VTR runs with the Sherman-Morrison update (Eqn 9.22 of Rich's RL book). Now code runs with dictionaries instead of numpy array, this means we can input state tuples as an index for our Q-values. As of right now, LSVI_UCB uses the same confidence bound as UCRL_VTR (Theorem 20.5 of Bandits Book). LSVI_UCB still needs to be optimized and further debugged.

agents.py - this file contains RLSVI, PSRL, UCRL_VTR, and UCBVI algorithms.

environments.py - this file contains the DeepSea and RiverSwim environments.



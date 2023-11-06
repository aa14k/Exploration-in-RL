# Exploration-in-RL
As of 11/06/23, EVILL, code for a new randomized algorithm that adds a randomized "regularizer" to the log-likelihood function in order to garuantee sample efficient learning in generalized linear bandits (and beyond). Examples include logistic and Rayleigh bandits!

As of 10/15/21, IDRL-VTR, code for a prototype algorithm that combines Information Directed Sampling with Value-Targeted Regression. Based off Dongrou's wonderful COLT paper with Quanquan and Csaba and Johannes' IDS work.

As of 5/15/21, [OPT-LSVI-PHE](https://arxiv.org/abs/2106.07841) was recently accepted to ICML2021!

As of 3/25/21, added a new folder called OPT-LSVI-PHE. This algorithm is an optimistic variant of PH-LSVI. This algorithm solves sparse mountain car very well. The code base was taken from the University of Alberta's Reinforcement Learning Coursera course, specially module 3 week 3. This algorithm can be thought of as an optimistic extension of RLSVI (Osbant et al, 2014). Interstingly, RLSVI (M=1) doesn't solve sparse mountain car but our algorithm (with M>1) does? 
As of 11/14/20, new code for Perturbed History (PH) Exploration for RL. This code compares PH-LSVI with RLSVI and LSVI-UCB on the episodic RiverSwim environment. Also some code for a paper that claims they have a TS algorithm that achieves same regret as UCB (whether this regret is correct is another question). Finally a new folder to test out exploration heuristics for RL, we propose a new method called BeyondGreedy exploration based on the probability matching of BeyondUCB (Foster and Raklin, 2020). 

As of 5/8/20, new code for UC-MatrixRL as been updated to the VTR_Paper_Code/Fixed_UC_VTR_Matrix.ipynb. DO NOT use other UC_MatrixRL code as it has not been properly debugged. I only kept the buggy code in this repo for documentation purposes. Also a new folder VTR_Paper_Code contains the code used to generate the figures in our VTR paper (Ayoub et al, 2020). Run the code in the Fixed_UC_VTR_Matrix.ipynb and once the code has finished executing and the data has been saved, open the two different plot scripts to recreate the plots in our paper! Finally, the code for LSVI-UCB (Jin et al, 2019) has been updated! Now with better confidence bounds as well as the option for epsilon-greedy exploration. Reproducability sure is EXCITING ! 

As of 2/20/20, only PSRL, RLSVI, UCRL_VTR, and UC_MatrixRL have been properly debugged and optimized to run on the riverswim environment. The other algorithms should work, though I have not spent as much time on them as the others.


PS: Feel free to email me any questions, suggestions, feedback, thoughts, etc..... (aayoub@ualberta.ca). 

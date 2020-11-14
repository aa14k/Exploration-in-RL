# Exploration-in-RL
As if 11/14/20, new code for Perturbed History Exploration for RL. This code compares PH-LSVI with RLSVI and LSVI-UCB on the episodic RiverSwim environment. 

As of 5/8/20, new code for UC-MatrixRL as been updated to the VTR_Paper_Code/Fixed_UC_VTR_Matrix.ipynb. DO NOT use other UC_MatrixRL code as it has not been properly debugged. I only kept the buggy code in this repo for documentation purposes. Also a new folder VTR_Paper_Code contains the code used to generate the figures in our VTR paper (Ayoub et al, 2020). Run the code in the Fixed_UC_VTR_Matrix.ipynb and once the code has finished executing and the data has been saved, open the two different plot scripts to recreate the plots in our paper! Finally, the code for LSVI-UCB (Jin et al, 2019) has been updated! Now with better confidence bounds as well as the option for epsilon-greedy exploration. Reproducability sure is EXCITING ! 

As of 2/20/20, only PSRL, RLSVI, UCRL_VTR, and UC_MatrixRL have been properly debugged and optimized to run on the riverswim environment. The other algorithms should work, though I have not spent as much time on them as the others.



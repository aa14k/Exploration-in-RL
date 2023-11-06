# Code for EVILL

The code in this repository can be used to reproduce the experimental results featuring in Figure 1 and Figure 2 of our paper,

> Janz, David, Shuai Liu, Alex Ayoub, and Csaba Szepesvári. “Exploration via linearly perturbed loss minimisation”. In: arxiv (to appear) (2023).

All these scripts write `.npz` files to the `results` folder. The included jupyter notebook, `evill_plots.ipynb`, may then be used to recreate the plots in the paper. First, however, we need create a `venv` and install the requisite packages. For this, run:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then the following four instructions should recreate all the necessary data. Running these should take no more than a couple minutes on a semi-modern laptop.

1. For the logistic bandit experiment, run the following for the high variance setting, left panel of Figure 1.

```
python3 run_logistic_bandit_experiment.py -n 10000 --lin_val_max 1.5 --lin_val_min 0.1 --param_decay 2.0 --output_name logistic_high_var
```

2. And the following for the low variance setting, right panel of Figure 1.

```
python3 run_logistic_bandit_experiment.py -n 10000 --lin_val_max 4.0 --lin_val_min 0.1 --param_decay 2.0 --output_name logistic_low_var
```

3. The following produces results for the estimation experiment, left panel of Figure 2. 

```
python3 run_rayleigh_estimation_experiment.py --output_name rayleigh_estimation
```

4. And for the bandit experiment, right panel of Figure 2, run:

```
python3 run_rayleigh_bandit_experiment.py --output_name rayleigh_bandit
```


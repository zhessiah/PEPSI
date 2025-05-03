# PEPSI

Open-source code for PEPSI: Pareto-Optimal Adversarial Regularization for Robust
Multi-agent Reinforcement Learning.


## Installation instructions

Install Python packages

```shell
# require Anaconda 3 or Miniconda 3
conda create -n PEPSI python=3.8 -y
conda activate PEPSI

bash install_dependecies.sh
```

Set up StarCraft II (2.4.10) and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

Set up Google Football:

```shell
bash install_gfootball.sh
```

## Command Line Tool

**Run an experiment**

```shell
# For SMAC
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m
```

```shell
# For Difficulty-Enhanced Predator-Prey
python3 src/main.py --config=qmix_predator_prey --env-config=stag_hunt with env_args.map_name=stag_hunt
```

```shell
# For Communication tasks
python3 src/main.py --config=maddpg --env-config=stag_hunt with env_args.map_name=stag_hunt
```

```shell
# For Google Football (Insufficient testing)
# map_name: academy_counterattack_easy, academy_counterattack_hard, five_vs_five...
python3 src/main.py --config=vdn_gfootball --env-config=gfootball with env_args.map_name=academy_counterattack_hard env_args.num_agents=4
```

The config files act as defaults for an algorithm or environment.

For robustness and adversarial attack details, go to `default.yaml` and change `Attack params`.


**Run n parallel experiments**

```shell
# 2 threads, on gpu 0, run 5 times each map

bash run.sh m3ddpg stag_hunt stag_hunt t_max=5050000 1 2 1

bash run.sh qmix mpe simple_spread epsilon_anneal_time=500000,td_lambda=0.3 1 0 1

bash run.sh qmix sc2 8m_vs_9m epsilon_anneal_time=500000,td_lambda=0.3 1 1 1

bash run.sh qatten gfootball academy_counterattack_easy epsilon_anneal_time=500000,td_lambda=0.3 1 1 1


```

`xxx_list` is separated by `,`.

All results will be stored in the `Results` folder and named with `map_name`.

**Kill all training processes**

```shell
# all python and game processes of current user will quit.
bash clean.sh
```

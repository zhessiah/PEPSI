#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
--config=shaq \
--env-config=sc2 with env_args.map_name=27m_vs_30m
#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PRIOR_TYPE=$1

for i in {0..23}
do
   python scripts/run_bootstrapped_exp.py --parallel_runs 24 --base_seed $i --prior PRIOR_TYPE --n_runs 4 &
   sleep 30
done


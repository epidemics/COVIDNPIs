#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

for i in 21 22
do
   echo "Running Experiment $i"
   python scripts/run_bugs_exp.py --exp $i &
   sleep 30
done


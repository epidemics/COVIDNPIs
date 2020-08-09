#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

for g in 0.15 0.175 0.2 0.225 0.25
do
  for a in 0.05 0.075 0.1 0.125 0.15
  do
     echo "Running Growth Noise $g region noise $a"
     python scripts/run_diffeff_crossval.py --g $g --a $a &
     sleep 30
   done
done


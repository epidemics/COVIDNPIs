#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

for gi in 0 1 2
do
  for cd in 0 1 2
  do
    for dd in 0 1 2
    do
        python scripts/run_epi_grid.py --cd $cd --dd $dd --gi $gi &
        sleep 30
    done
   done
done
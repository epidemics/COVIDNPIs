#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

for g in 0.001, 0.05, 0.1
do
  for a in 0.05 0.075 0.1 0.125 0.15
  do
    if [ $a!="0.15" ] && [ $g!="0.25" ];then
        echo "Running Growth Noise $g region noise $a"
        python scripts/run_difeff_crossval.py --g $g --a $a &
        sleep 30
    fi
   done
done

python scripts/run_difeff_crossval.py --g 0.25 --a 0.15 &
#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python scripts/extra_exps.py --e 0 &
python scripts/extra_exps.py --e 1 &
python scripts/extra_exps.py --e 2 &
python scripts/extra_exps.py --e 3 &
python scripts/extra_exps.py --e 4 &
python scripts/extra_exps.py --e 5 &

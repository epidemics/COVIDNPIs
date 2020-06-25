#!/bin/bash

python scripts/run_dif_dates.py --m 2 --l 0 &
python scripts/run_dif_dates.py --m 3 --l 0 &
python scripts/run_dif_dates.py --m 4 --l 0 &
python scripts/run_dif_dates.py --m 2 --l 2 &
python scripts/run_dif_dates.py --m 3 --l 2 &
python scripts/run_dif_dates.py --m 4 --l 2 &
python scripts/run_dif_dates.py --m 2 --l 4 &
python scripts/run_dif_dates.py --m 3 --l 4 &
python scripts/run_dif_dates.py --m 4 --l 4 &


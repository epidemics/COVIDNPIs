#!/bin/bash

python scripts/run_dif_dates.py --m 0 --l 1 &
python scripts/run_dif_dates.py --m 0 --l 4 &
python scripts/run_dif_dates.py --m 1 --l 1 &
python scripts/run_dif_dates.py --m 1 --l 4 &

#!/bin/bash
#!/bin/bash
for g in 0.1 0.2 0.3
for h in 0.05 0.1 0.15
do
     echo "running with noise $g and $h"
     python run_g_exp_v5.py --g $g --h $h --s 2000 --c 4&
done